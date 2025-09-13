#include <stdio.h>
#include <fcntl.h> // for creat()
#include <unistd.h> // for close()
#include <errno.h> //for errno
#include <string.h> //for strerror()

int main(int argc, char **argv){
    if(argc != 2){
        printf("Error \nUsage: %s <filename>\n", argv[0]);
        return 1;
    }

     // Try to create a file with rwxr-xr-x (0755)
     int fd = creat(argv[1], 0755);
     if (fd<0){
        if(errno == EEXIST){
         printf("Error: File '%s' already exists! (errno=%d: %s)\n",argv[1],errno, strerror(errno));
        }else{
            perror("creat failed");
        }
         return 2;
        }
        printf("File '%s' created successfully with mode 0755 (subject to umask).\n", argv[1]);
     close(fd);
     return 0;
}
