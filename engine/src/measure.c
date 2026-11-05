// POSIX process measurement. Compilation is outside the measured interval.
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: measure output_prefix command [arguments ...]\n");
    return 2;
  }
  size_t size = strlen(argv[1]) + 32;
  char *path = malloc(size);
  if (!path)
    return 2;
  snprintf(path, size, "%s.csv", argv[1]);
  int out = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (out < 0) {
    perror(path);
    free(path);
    return 2;
  }
  snprintf(path, size, "%s.log", argv[1]);
  int err = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (err < 0) {
    perror(path);
    close(out);
    free(path);
    return 2;
  }
  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 2;
  }
  if (pid == 0) {
    if (dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0)
      _exit(126);
    close(out);
    close(err);
    execvp(argv[2], argv + 2);
    perror("execvp");
    _exit(127);
  }
  close(out);
  close(err);
  int status;
  struct rusage usage;
  pid_t got;
  do {
    got = wait4(pid, &status, 0, &usage);
  } while (got < 0 && errno == EINTR);
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (got < 0) {
    perror("wait4");
    return 2;
  }
  double seconds =
      end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) * 1e-9;
  unsigned long long peak = (unsigned long long)usage.ru_maxrss;
#ifndef __APPLE__
  peak *= 1024ULL;
#endif
  int result = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  snprintf(path, size, "%s.resources.csv", argv[1]);
  int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) {
    perror(path);
    return 2;
  }
  FILE *f = fdopen(fd, "w");
  if (!f) {
    perror("fdopen");
    close(fd);
    return 2;
  }
  fprintf(f, "wall_seconds,peak_rss_bytes,exit_code\n%.9f,%llu,%d\n", seconds,
          peak, result);
  if (fclose(f)) {
    perror("resource output");
    return 2;
  }
  free(path);
  return result;
}
