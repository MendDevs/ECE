#include<unistd.h>
#include<sys/types.h>
#include<stdio.h>

int main(){

   /*
   pid_t pid = fork();

   if (pid == 0) printf("PID fils -> %d\n",getpid());
   else  if (pid > 0) printf("PID pere -> %d\n", getppid());
   else printf("error");
   */

   /*
       pid_t p ;
  int a = 20;
  switch (p = fork()){
  case -1:
      perror("le fork a echoue !" ) ; break ;
    case 0 :
      printf("ici processus fils , le PID%d.\n", getpid());
      a+=10;
     break ;
    default :
      printf("ici processus pere, le PID%d.\n", getpid());
      a += 100;
  }
  printf("Fin du processus%d avec a=%d.\n", getpid(), a);
  return 0;
  */

   pid_t p;
   int a = 20;
   switch (p = fork())
   {
   case -1:
       perror("le fork a echoue !");
       break;
   case 0:
       printf("ici processus fils , le PID%d.\n", getpid());
       a += 10;
       break;
   default:
       printf("ici processus pere, le PID%d.\n", getpid());
       a += 100;
   }
   printf("Fin du processus%d avec a=%d.\n", getpid(), a);
   return 0;
   }