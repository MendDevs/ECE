#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>

int main(void){
    int fd;
    char message[26];
    sprintf(message, "Bonjour du writer [%d]\n",getpid());
    fd=open("mypipe", O_RDONLY);
    printf("ici reader[%d] \n", getpid());
    if (fd!=1){
        printf("Recu par le lecteur: \n");
        whiile((n = read(fd,&input, 1)) > 0){
            printf("%", input);
        }
        printf("Le lecteur se termine!\n",n);
    }
    else
        printf("Desole, le tube n'est pas disponible \n");
    close(fd);
    return 0;
}
