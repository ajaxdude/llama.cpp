#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/sched.h>
#include <sched.h>
#include <poll.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if !defined(SYS_clone3)
#define SYS_clone3 435
#endif
#if !defined(SYS_pidfd_send_signal)
#define SYS_pidfd_send_signal 424
#endif
#if !defined(SYS_pidfd_open)
#define SYS_pidfd_open 434
#endif

extern char ** environ;
#endif

#ifndef DSV41_BUILD_REVISION
#define DSV41_BUILD_REVISION "unknown"
#endif

namespace {

struct options {
    int protocol_fd = -1;
    int expected_parent = -1;
    std::string exec_path;
    std::vector<int> keep_fds;
    std::vector<char *> target_argv;
};

int parse_positive_int(const char * value, const char * label) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<int>(parsed);
}

options parse_options(int argc, char ** argv) {
    options result;
    int index = 1;
    while (index < argc && std::strcmp(argv[index], "--") != 0) {
        if (std::strcmp(argv[index], "--protocol-fd") == 0 && index + 1 < argc) {
            result.protocol_fd = parse_positive_int(argv[index + 1], "protocol descriptor");
            index += 2;
        } else if (std::strcmp(argv[index], "--expected-parent") == 0 && index + 1 < argc) {
            result.expected_parent = parse_positive_int(argv[index + 1], "expected parent");
            index += 2;
        } else if (std::strcmp(argv[index], "--exec-path") == 0 && index + 1 < argc) {
            result.exec_path = argv[index + 1];
            index += 2;
        } else if (std::strcmp(argv[index], "--keep-fd") == 0 && index + 1 < argc) {
            result.keep_fds.push_back(parse_positive_int(argv[index + 1], "retained descriptor"));
            index += 2;
        } else {
            throw std::runtime_error("unknown containment helper argument");
        }
    }
    if (index >= argc || std::strcmp(argv[index], "--") != 0) {
        throw std::runtime_error("containment helper target separator is missing");
    }
    ++index;
    if (result.protocol_fd < 0 || result.expected_parent < 0 || result.exec_path.empty() ||
            index >= argc) {
        throw std::runtime_error("containment helper arguments are incomplete");
    }
    for (; index < argc; ++index) {
        result.target_argv.push_back(argv[index]);
    }
    result.target_argv.push_back(nullptr);
    return result;
}

#if defined(__linux__)

bool retained_fd(int fd, const std::vector<int> & keep_fds, int extra_fd) {
    if (fd >= 0 && fd <= STDERR_FILENO) {
        return true;
    }
    if (fd == extra_fd) {
        return true;
    }
    for (int keep_fd : keep_fds) {
        if (fd == keep_fd) {
            return true;
        }
    }
    return false;
}

void close_checked(int fd, const char * label) {
    if (close(fd) != 0) {
        throw std::runtime_error(std::string("cannot close ") + label);
    }
}

void close_unneeded_fds(const std::vector<int> & keep_fds, int extra_fd) {
    DIR * directory = opendir("/proc/self/fd");
    if (directory == nullptr) {
        throw std::runtime_error("cannot open procfs descriptor directory");
    }
    const int directory_fd = dirfd(directory);
    while (dirent * entry = readdir(directory)) {
        char * end = nullptr;
        const long parsed = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0' || parsed < 0 ||
                parsed > std::numeric_limits<int>::max()) {
            continue;
        }
        const int fd = static_cast<int>(parsed);
        if (fd != directory_fd && !retained_fd(fd, keep_fds, extra_fd)) {
            close_checked(fd, "unneeded descriptor");
        }
    }
    if (closedir(directory) != 0) {
        throw std::runtime_error("cannot close procfs descriptor directory");
    }
}

void require_initial_signal_state() {
    sigset_t mask;
    if (sigprocmask(SIG_SETMASK, nullptr, &mask) != 0) {
        throw std::runtime_error("cannot read containment helper signal mask");
    }
    for (int signal_number = 1; signal_number < NSIG; ++signal_number) {
        if (signal_number == SIGKILL || signal_number == SIGSTOP) {
            continue;
        }
        struct sigaction action {};
        if (sigaction(signal_number, nullptr, &action) != 0) {
            if (errno == EINVAL) {
                continue;
            }
            throw std::runtime_error("cannot read containment helper signal disposition");
        }
        if (sigismember(&mask, signal_number) != 1) {
            throw std::runtime_error("containment helper inherited an unblocked signal");
        }
        if (action.sa_handler != SIG_DFL) {
            throw std::runtime_error("containment helper inherited a signal handler");
        }
    }
}

void unblock_default_signals() {
    sigset_t empty;
    sigemptyset(&empty);
    if (sigprocmask(SIG_SETMASK, &empty, nullptr) != 0) {
        throw std::runtime_error("cannot unblock containment helper signals");
    }
}

void set_parent_death(pid_t expected_parent) {
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
        throw std::runtime_error("cannot set containment parent-death signal");
    }
    if (getppid() != expected_parent) {
        throw std::runtime_error("containment parent changed before lifecycle binding");
    }
}

void write_all(int fd, const void * data, size_t size) {
    const char * cursor = static_cast<const char *>(data);
    while (size > 0) {
        const ssize_t written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("containment protocol write failed");
        }
        cursor += written;
        size -= static_cast<size_t>(written);
    }
}

void write_mapping_file(int process_directory, const char * name, const std::string & content) {
    const int fd = openat(process_directory, name, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        throw std::runtime_error(std::string("cannot open namespace ") + name);
    }
    write_all(fd, content.data(), content.size());
    close_checked(fd, "namespace mapping descriptor");
}

std::string receive_packet(int fd) {
    char buffer[128];
    const ssize_t size = recv(fd, buffer, sizeof(buffer), 0);
    if (size <= 0) {
        throw std::runtime_error("containment protocol closed");
    }
    return std::string(buffer, static_cast<size_t>(size));
}

void send_packet(int fd, const char * packet) {
    const size_t size = std::strlen(packet);
    if (send(fd, packet, size, MSG_NOSIGNAL) != static_cast<ssize_t>(size)) {
        throw std::runtime_error("containment protocol send failed");
    }
}

void send_pidfd(int fd, int pidfd) {
    char payload[] = "PREPARED";
    iovec vector {payload, sizeof(payload) - 1};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] {};
    msghdr message {};
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    cmsghdr * header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(header), &pidfd, sizeof(pidfd));
    if (sendmsg(fd, &message, MSG_NOSIGNAL) != static_cast<ssize_t>(sizeof(payload) - 1)) {
        throw std::runtime_error("cannot send namespace pidfd");
    }
}

bool pidfd_has_exited(int pidfd) {
    pollfd descriptor {pidfd, POLLIN, 0};
    const int result = poll(&descriptor, 1, 0);
    if (result < 0) {
        throw std::runtime_error("cannot query containment pidfd");
    }
    return result > 0 && (descriptor.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
}

int wait_status_exit_code(int status) {
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 125;
}

class namespace_owner {
public:
    namespace_owner(pid_t pid, int pidfd) : pid_(pid), pidfd_(pidfd) {
    }

    ~namespace_owner() {
        if (reaped_) {
            return;
        }
        if (pidfd_ >= 0) {
            syscall(SYS_pidfd_send_signal, pidfd_, SIGKILL, nullptr, 0);
        }
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
        }
        if (pidfd_ >= 0) {
            close(pidfd_);
        }
    }

    int pidfd() const {
        return pidfd_;
    }

    int wait() {
        int status = 0;
        while (waitpid(pid_, &status, 0) < 0) {
            if (errno != EINTR) {
                throw std::runtime_error("cannot reap target PID namespace");
            }
        }
        reaped_ = true;
        return status;
    }

    void close_pidfd() {
        const int descriptor = pidfd_;
        pidfd_ = -1;
        if (close(descriptor) != 0) {
            throw std::runtime_error("cannot close namespace pidfd");
        }
    }

private:
    pid_t pid_;
    int pidfd_;
    bool reaped_ = false;
};

[[noreturn]] void run_namespace_init(
        int release_fd,
        int ready_fd,
        int mapping_fd,
        int protocol_fd,
        int helper_pidfd,
        const options & config) {
    try {
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
            _exit(125);
        }
        if (getppid() != 0 || pidfd_has_exited(helper_pidfd)) {
            _exit(125);
        }
        close_checked(helper_pidfd, "namespace helper pidfd");
        close_checked(protocol_fd, "namespace protocol descriptor");
        write_all(ready_fd, "B", 1);
        char mapped = 0;
        ssize_t mapped_size;
        do {
            mapped_size = read(mapping_fd, &mapped, 1);
        } while (mapped_size < 0 && errno == EINTR);
        if (mapped_size != 1 || mapped != 'M') {
            _exit(125);
        }
        close_checked(mapping_fd, "namespace mapping descriptor");
        if (setresgid(0, 0, 0) != 0 || setresuid(0, 0, 0) != 0) {
            _exit(125);
        }
        if (mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0 ||
                umount2("/proc", MNT_DETACH) != 0 ||
                mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, nullptr) != 0) {
            _exit(125);
        }
        write_all(ready_fd, "R", 1);
        close_checked(ready_fd, "namespace readiness descriptor");
        char release = 0;
        ssize_t received;
        do {
            received = read(release_fd, &release, 1);
        } while (received < 0 && errno == EINTR);
        if (received != 1 || release != 'X') {
            _exit(125);
        }
        close_checked(release_fd, "namespace release descriptor");
        const pid_t target_pid = fork();
        if (target_pid < 0) {
            _exit(125);
        }
        if (target_pid == 0) {
            for (int signal_number = 1; signal_number < NSIG; ++signal_number) {
                if (signal_number == SIGKILL || signal_number == SIGSTOP) {
                    continue;
                }
                struct sigaction action {};
                action.sa_handler = SIG_DFL;
                sigemptyset(&action.sa_mask);
                sigaction(signal_number, &action, nullptr);
            }
            sigset_t empty;
            sigemptyset(&empty);
            sigprocmask(SIG_SETMASK, &empty, nullptr);
            close_unneeded_fds(config.keep_fds, -1);
            execve(config.exec_path.c_str(), config.target_argv.data(), environ);
            _exit(127);
        }
        int target_status = 0;
        while (waitpid(target_pid, &target_status, 0) < 0) {
            if (errno != EINTR) {
                _exit(125);
            }
        }
        kill(-1, SIGKILL);
        while (waitpid(-1, nullptr, 0) >= 0 || errno == EINTR) {
        }
        _exit(wait_status_exit_code(target_status));
    } catch (...) {
        _exit(125);
    }
}

int run_linux_helper(options config) {
    require_initial_signal_state();
    set_parent_death(config.expected_parent);
    close_unneeded_fds(config.keep_fds, config.protocol_fd);
    unblock_default_signals();
    send_packet(config.protocol_fd, "READY");
    if (receive_packet(config.protocol_fd) != "PREPARE") {
        throw std::runtime_error("containment PREPARE packet is invalid");
    }

    int release_pipe[2] {-1, -1};
    if (pipe2(release_pipe, O_CLOEXEC) != 0) {
        throw std::runtime_error("cannot create namespace release pipe");
    }
    int ready_pipe[2] {-1, -1};
    if (pipe2(ready_pipe, O_CLOEXEC) != 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        throw std::runtime_error("cannot create namespace readiness pipe");
    }
    int mapping_pipe[2] {-1, -1};
    if (pipe2(mapping_pipe, O_CLOEXEC) != 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        throw std::runtime_error("cannot create namespace mapping pipe");
    }
    const int helper_pidfd = static_cast<int>(syscall(SYS_pidfd_open, getpid(), 0));
    if (helper_pidfd < 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        close(mapping_pipe[0]);
        close(mapping_pipe[1]);
        throw std::runtime_error("cannot open stable helper identity");
    }
    int namespace_pidfd = -1;
    clone_args arguments {};
    arguments.flags = CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNS | CLONE_PIDFD;
    arguments.pidfd = reinterpret_cast<uintptr_t>(&namespace_pidfd);
    arguments.exit_signal = SIGCHLD;
    const pid_t namespace_init = static_cast<pid_t>(
        syscall(SYS_clone3, &arguments, sizeof(arguments)));
    if (namespace_init < 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        close(mapping_pipe[0]);
        close(mapping_pipe[1]);
        close(helper_pidfd);
        throw std::runtime_error(std::string("cannot create target PID namespace: ") + std::strerror(errno));
    }
    if (namespace_init == 0) {
        close_checked(release_pipe[1], "namespace release writer");
        close_checked(ready_pipe[0], "namespace readiness reader");
        close_checked(mapping_pipe[1], "namespace mapping writer");
        run_namespace_init(
            release_pipe[0], ready_pipe[1], mapping_pipe[0],
            config.protocol_fd, helper_pidfd, config);
    }
    namespace_owner owned_namespace(namespace_init, namespace_pidfd);
    close_checked(release_pipe[0], "helper release reader");
    close_checked(ready_pipe[1], "helper readiness writer");
    close_checked(mapping_pipe[0], "helper mapping reader");
    char ready = 0;
    ssize_t ready_size;
    do {
        ready_size = read(ready_pipe[0], &ready, 1);
    } while (ready_size < 0 && errno == EINTR);
    if (ready_size != 1 || ready != 'B') {
        throw std::runtime_error("target PID namespace did not bind helper lifetime");
    }
    const std::string process_path = "/proc/" + std::to_string(namespace_init);
    const int process_directory = open(
        process_path.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (process_directory < 0 || pidfd_has_exited(namespace_pidfd)) {
        if (process_directory >= 0) {
            close_checked(process_directory, "namespace process directory");
        }
        throw std::runtime_error("cannot retain target namespace mapping identity");
    }
    write_mapping_file(process_directory, "setgroups", "deny\n");
    write_mapping_file(
        process_directory, "uid_map",
        "0 " + std::to_string(getuid()) + " 1\n");
    write_mapping_file(
        process_directory, "gid_map",
        "0 " + std::to_string(getgid()) + " 1\n");
    close_checked(process_directory, "namespace process directory");
    write_all(mapping_pipe[1], "M", 1);
    close_checked(mapping_pipe[1], "helper mapping writer");
    do {
        ready_size = read(ready_pipe[0], &ready, 1);
    } while (ready_size < 0 && errno == EINTR);
    close_checked(ready_pipe[0], "helper readiness reader");
    close_checked(helper_pidfd, "helper self pidfd");
    if (ready_size != 1 || ready != 'R') {
        throw std::runtime_error("target PID namespace setup did not complete");
    }
    send_pidfd(config.protocol_fd, owned_namespace.pidfd());
    if (receive_packet(config.protocol_fd) != "EXEC") {
        throw std::runtime_error("containment EXEC packet is invalid");
    }
    write_all(release_pipe[1], "X", 1);
    close_checked(release_pipe[1], "helper release writer");
    send_packet(config.protocol_fd, "RELEASED");
    const int status = owned_namespace.wait();
    owned_namespace.close_pidfd();
    send_packet(config.protocol_fd, "COMPLETE");
    return wait_status_exit_code(status);
}

#endif

}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
            std::cout << "deepseek-v41-containment-helper " << DSV41_BUILD_REVISION << '\n';
            return 0;
        }
#if defined(__linux__)
        return run_linux_helper(parse_options(argc, argv));
#else
        (void) argc;
        (void) argv;
        std::cerr << "Linux PID namespace containment is unavailable on this platform\n";
        return 125;
#endif
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 125;
    }
}
