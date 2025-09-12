#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<unistd.h>
#include<pwd.h>
#include<grp.h>
#include<fcntl.h>


 int main(int argc, char **argv){
    //check if exactly 1 argument (filename) is given
    if(argc !=2){
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    struct stat fileInfo;

    //open the file first (needed for fstat)
    int fd = open(argv[1], O_RDONLY);
    if(fd <0){
            perror("Open failed");
            return 1;
    }

    // Get file info using fstat
    if(fstat(fd,&fileInfo)<0){
        perror("fstat failed");
        close(fd);
        return 1;
    }

    printf("[fstat] File Info Of %s\n", argv[1]);
    printf("-------------------------------------\n");

    //1. Inode number
    printf("Inode Number: \t\t%lu\n", (unsigned long)fileInfo.st_ino);

    //2. #of link
    printf("Total Links: \t\t%lu\n", (unsigned long) fileInfo.st_nlink);

    //3. Owner (user)
    struct passwd *pw = getpwuid(fileInfo.st_uid);
    if(pw != NULL)
        printf("Owner: \t\t\t%s\n", pw->pw_name);
    else
        printf("Owner UID: \t\t%d\n", fileInfo.st_uid);

    //4. Group
    struct group *grp = getgrgid(fileInfo.st_gid);
    printf("Group: \t\t\t%s\n", grp ? grp->gr_name : "Unknown");

    //5. Size
    printf("Size : \t\t\t%ld\n", (long) fileInfo.st_size);

    //6. File Type
    printf("Type: \t\t\t");
    if (S_ISREG(fileInfo.st_mode)) printf("Regular file \n");
    else if (S_ISDIR(fileInfo.st_mode)) printf("Regular file \n");
    else if (S_ISLNK(fileInfo.st_mode)) printf("Directory \n");

    else if (S_ISCHR(fileInfo.st_mode)) printf("Character device\n");
    else if (S_ISBLK(fileInfo.st_mode)) printf("Block device\n");
    else if (S_ISFIFO(fileInfo.st_mode)) printf("FIFO/pipe\n");
    else if (S_ISSOCK(fileInfo.st_mode)) printf("Socket\n");
    else printf("Unknown\n");

    close(fd);
    return 0;
}
