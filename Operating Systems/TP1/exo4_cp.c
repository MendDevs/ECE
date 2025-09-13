#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <libgen.h>
#include <errno.h>

#define BUF_SZ 8192

static void die_perror(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <source> <destination>\n", argv[0]);
        return 2;
    }

    const char *src_path = argv[1];
    const char *dst_path = argv[2];

    /* Stat source */
    struct stat st_src;
    if (stat(src_path, &st_src) < 0) {
        perror("stat(source)");
        return 2;
    }

    /* If source is a directory, refuse (this simple cp handles files only) */
    if (S_ISDIR(st_src.st_mode)) {
        fprintf(stderr, "Source is a directory: %s\n", src_path);
        return 1;
    }

    /* If destination is a directory, append basename(source) */
    struct stat st_dst;
    char *dst_final = NULL;
    if (stat(dst_path, &st_dst) == 0 && S_ISDIR(st_dst.st_mode)) {
        /* build path: dst_path / basename(src_path) */
        char *src_copy = strdup(src_path);
        if (!src_copy) die_perror("strdup");
        char *base = basename(src_copy); /* note: basename may modify its arg */
        size_t len = strlen(dst_path) + 1 + strlen(base) + 1;
        dst_final = malloc(len);
        if (!dst_final) die_perror("malloc");
        snprintf(dst_final, len, "%s/%s", dst_path, base);
        free(src_copy);
    } else {
        /* destination is not an existing directory; use dst_path as final */
        dst_final = strdup(dst_path);
        if (!dst_final) die_perror("strdup");
    }

    /* If destination exists, stat it and check if it is the same inode as source */
    if (stat(dst_final, &st_dst) == 0) {
        if (st_src.st_dev == st_dst.st_dev && st_src.st_ino == st_dst.st_ino) {
            fprintf(stderr, "Source and destination are the same file: %s\n", dst_final);
            free(dst_final);
            return 1;
        }
    }

    /* Open source for reading */
    int fd_src = open(src_path, O_RDONLY);
    if (fd_src < 0) {
        perror("open(source)");
        free(dst_final);
        return 2;
    }

    /* Create/truncate destination. Use permissions from source, but umask may apply.
       We'll call fchmod afterwards to ensure exact permission bits. */
    mode_t mode = st_src.st_mode & 07777;
    int fd_dst = open(dst_final, O_WRONLY | O_CREAT | O_TRUNC, mode);
    if (fd_dst < 0) {
        perror("open(destination)");
        close(fd_src);
        free(dst_final);
        return 2;
    }

    /* Ensure destination has exact permissions (override umask effect) */
    if (fchmod(fd_dst, mode) < 0) {
        /* Not fatal for copying, but warn */
        perror("fchmod");
    }

    /* Buffered copy with correct handling of partial writes */
    ssize_t r;
    char buf[BUF_SZ];
    while ((r = read(fd_src, buf, sizeof(buf))) > 0) {
        ssize_t written = 0;
        while (written < r) {
            ssize_t w = write(fd_dst, buf + written, r - written);
            if (w < 0) {
                if (errno == EINTR) continue;
                perror("write");
                close(fd_src);
                close(fd_dst);
                free(dst_final);
                return 3;
            }
            written += w;
        }
    }
    if (r < 0) {
        perror("read");
        close(fd_src);
        close(fd_dst);
        free(dst_final);
        return 3;
    }

    /* Close files */
    if (close(fd_src) < 0) perror("close(source)");
    if (close(fd_dst) < 0) perror("close(destination)");

    free(dst_final);
    return 0;
}
