#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t pid1, pid2, pid3;

    if ((pid1 = fork()) == 0) {
        printf("Fils1 (pid=%d) : je deviens zombie puis orphelin\n", getpid());
        exit(0); // zombie → père ne fait pas wait()
    }

    if ((pid2 = fork()) == 0) {
        printf("Fils2 (pid=%d) : je deviens orphelin puis zombie\n", getpid());
        sleep(5); // père meurt avant
        exit(0);
    }

    if ((pid3 = fork()) == 0) {
        printf("Fils3 (pid=%d) : je deviens zombie mais pas orphelin\n", getpid());
        exit(0);
    }

    sleep(2);
    printf("Père (pid=%d) : je termine sans wait pour fils1 et fils3.\n", getpid());
    exit(0);
}
