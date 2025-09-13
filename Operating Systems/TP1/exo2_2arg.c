#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<unistd.h>
#include<pwd.h>
#include<grp.h>
#include<fcntl.h>

int main(int argc[],char **argv){

        //incorrect number of parameters
        if(argc !=3 ){
            printf("Error: Incorrect number of parameters");
            printf("Usage: %s <file1 name> <file2 name>",argv[0]);
            return -1;
        }

        struct stat file1,file2;

        //primitive stat error
        if((stat(argv[1], &file1) < 0 )|| (stat(argv[2], &file2)<0)){
            perror("Stat Error");
            return 2;
        }

        //same node
        else if (file1.st_ino==file2.st_ino &&
            file1.st_dev==file2.st_dev){
            printf("Identical files");
            return 1;
        }else{
            //different nodes
            printf("Inode number of file 1: \t%d\n",file1.st_ino);
            printf("Inode number of file 2: \t%d\n",file2.st_ino);
            return 0;
        }

    return 0;
}
