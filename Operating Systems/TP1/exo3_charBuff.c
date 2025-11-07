#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#define BUF_SIZE 1024 // buffer size

int main(int argc, char **argv){
    if(argc != 2){
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    // Open source file
    int fd = open(argv[1], O_RDONLY);
    if(fd < 0){
        perror("open failed");
        return 1;
    }

    // Get file info
    struct stat fileInfo;
    if(fstat(fd, &fileInfo) < 0){
        perror("fstat failed");
        close(fd);
        return 1;
    }

    // Check if it's a regular file
    if(!S_ISREG(fileInfo.st_mode)){
        printf("Not a regular file!\n");
        close(fd);
        return 1;
    }

    ////////////////////// VERSION 1: char by char //////////////////////

    char c;
    while (read(fd, &c, 1) == 1){
        write(STDOUT_FILENO, &c, 1);
    }


    ////////////////////// VERSION 2: buffer (1024) //////////////////////
    /*
    char buffer[BUF_SIZE];
    ssize_t bytesRead;
    while((bytesRead = read(fd, buffer, BUF_SIZE)) > 0){
        write(STDOUT_FILENO, buffer, bytesRead);
    }
    */

    close(fd);
    return 0;

  /*
    Result after using /bin/time:
    char-by-char: much slower (because each read and write is done one character at a time → many syscalls).
    buffer (1024): much faster (fewer syscalls because you handle 1024 bytes at once).

  */
}
