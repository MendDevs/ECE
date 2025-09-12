#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char **argv){

   if(argc !=2) return 1;

   struct stat fileInfo;
   if(stat(argv[1], &fileInfo)<0){
        perror("stat failed");
        return 1;
   }



return 0;
}
