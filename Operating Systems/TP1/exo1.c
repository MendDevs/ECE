#include <stdio.h>
#include <sys/stat.h>
#include<sys/types.h>
#include<unistd.h> //for calling low level unix functions
#include<pwd.h>  //for getting user name
#include<grp.h>  //for getting group name

int main(int argc, char **argv){
    //counts the number of arguments

    //check if exactly 1 argument (filename) is given
   if(argc !=2){
        printf("Usage: %s<filename>\n",argv[0]);
    return 1;}

   struct stat fileInfo;

   //Get file info
   if(stat(argv[1], &fileInfo)<0){
        perror("stat failed");
        return 1;
   }

    printf("[stat] File Info of %s\n" , argv[1]);
    printf("--------------------------------------\n");

    //1. Inode number
    printf("Inode Number: \t\t%d\n", fileInfo.st_ino);

    //2. #of link
    printf("Total Links : \t\t%d\n", fileInfo.st_nlink);

    //3. Owner (user)
    struct passwd *pw = getpwuid(fileInfo.st_uid);
    if (pw != NULL)
        printf("Owner: \t\t\t%s\n", pw->pw_name);
    else
        printf("Owner UID: \t\t%d\n", fileInfo.st_uid);

    //4. Group
    struct  group *grp = getgrgid(fileInfo.st_gid);
    if(grp != NULL)
        printf("Group GID: \t\t%d\n", grp->gr_name);
    else
        printf("Group GID: \t\t%d\n", fileInfo.st_gid);

    //5. Size


    //6. File Type

    return 0;
}

}
