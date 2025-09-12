#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char **argv){
    //counts the number of arguments

   if(argc !=2) return 1;

   struct stat fileInfo;
   if(stat(argv[1], &fileInfo)<0){
        perror("stat failed");
        return 1;
   }

    printf("[stat] File Info of %s\n" , argv[1]);
    printf("--------------------------------------\n");

    printf("Inode Number: \t\t%d\n", fileInfo.st_ino); //1. Inode number
    printf("Total Links : \t\t%d\n", fileInfo.st_nlink); //2. #of link

    printf("");//its owners
    //groups
    //size
    //type

return 0;
}
