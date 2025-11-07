#include <stdio.h>
#include<signal.h>
#include<unistd.h>

void sighandler(int signum)
{
    printf("Masquage du signal SIGTERM\n");
}

int main(void){
    char buffer[256];
    if(signal(SIGTERM,&sighandler)== SIG_ERR)
    {
        printf("Ne peut pas manipuler le signal'n");
        exit(1);
    }
    while(1){
        fgets(buffer,sizeof(buffer), stdin);
        printf("Input: %s\n",buffer);
    }

}

