
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    for (int i = 0; i < 3; i++) {
        if (fork() == 0) {
            printf("Fils %d terminé\n", i+1);
            exit(0);
        }
    }

    sleep(1); // laisser les fils devenir zombies

    int pid;
    while ((pid = waitpid(-1, NULL, WNOHANG)) > 0) {
        printf("Zombie %d nettoyé\n", pid);
    }
    return 0;
}
