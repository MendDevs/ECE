
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <grp.h>
#include <time.h>
#include <fcntl.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    struct stat fileInfo;

    // Version avec stat()
    if (stat(argv[1], &fileInfo) < 0) {
        perror("stat failed");
        return 2;
    }

    printf("[stat] File Info of %s\n", argv[1]);
    printf("-----------------------------------\n");

    printf("i-node number     : %lu\n", (unsigned long) fileInfo.st_ino);
    printf("Size (bytes)      : %ld\n", (long) fileInfo.st_size);
    printf("Permissions       : %o (octal)\n", fileInfo.st_mode & 0777);
    printf("Block size        : %ld\n", (long) fileInfo.st_blksize);
    printf("Links count       : %ld\n", (long) fileInfo.st_nlink);
    printf("Blocks allocated  : %ld\n", (long) fileInfo.st_blocks);
    printf("Owner ID (UID)    : %d\n", fileInfo.st_uid);
    printf("Group ID (GID)    : %d\n", fileInfo.st_gid);
    printf("Last access time  : %s", ctime(&fileInfo.st_atime));

    // ------- Version fstat -------
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open failed");
        return 3;
    }

    if (fstat(fd, &fileInfo) < 0) {
        perror("fstat failed");
        close(fd);
        return 4;
    }

    printf("\n[fstat] Same Info using fstat()\n");
    printf("-----------------------------------\n");
    printf("i-node number     : %lu\n", (unsigned long) fileInfo.st_ino);
    printf("Size (bytes)      : %ld\n", (long) fileInfo.st_size);
    printf("Permissions       : %o (octal)\n", fileInfo.st_mode & 0777);
    printf("Block size        : %ld\n", (long) fileInfo.st_blksize);
    printf("Links count       : %ld\n", (long) fileInfo.st_nlink);
    printf("Blocks allocated  : %ld\n", (long) fileInfo.st_blocks);
    printf("Owner ID (UID)    : %d\n", fileInfo.st_uid);
    printf("Group ID (GID)    : %d\n", fileInfo.st_gid);
    printf("Last access time  : %s", ctime(&fileInfo.st_atime));

    close(fd);
    return 0;
}
