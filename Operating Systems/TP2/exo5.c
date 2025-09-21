#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>

#define MaxJoueurs 4

void jouer(int NumJoueur) {
    printf("Joueur %d (pid=%d) joue\n", NumJoueur, getpid());
    sleep(2);
    printf("Joueur %d terminé\n", NumJoueur);
    exit(0);
}

int main() {
    pid_t pid[MaxJoueurs];

    // créer MaxJoueurs
    for (int i = 0; i < MaxJoueurs; i++) {
        if ((pid[i] = fork()) == 0) {
            jouer(i);
        }
    }

    while (1) {
        int status;
        pid_t p = wait(&status); // attendre un joueur
        if (p < 0) break; // plus de joueurs

        // retrouver le numéro du joueur terminé
        for (int i = 0; i < MaxJoueurs; i++) {
            if (pid[i] == p) {
                if ((pid[i] = fork()) == 0) {
                    jouer(i);
                }
            }
        }
    }
    return 0;
}

