#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void afficher(int n, char lettre)
{
    int i, j;
    for (j = 1; j < n; j++)
    {
        for (i = 1; i < 10000000; i++)
            ; // simulate delay
        printf("%c", lettre);
        fflush(stdout);
    }
}

void *threadA(void *inutilise)
{
    afficher(100, 'A');
    printf("\nFin du thread A\n");
    fflush(stdout);
    pthread_exit(NULL);
}

void *threadC(void *inutilise)
{
    afficher(150, 'C');
    printf("\nFin du thread C\n");
    fflush(stdout);
    pthread_exit(NULL);
}

void *threadB(void *inutilise)
{
    pthread_t thC;
    pthread_create(&thC, NULL, threadC, NULL);
    afficher(100, 'B');
    printf("\nLe thread B attend la fin du thread C\n");
    pthread_join(thC, NULL);
    printf("\nFin du thread B\n");
    fflush(stdout);
    pthread_exit(NULL);
}

int main()
{
    pthread_t thA, thB;

    printf("Création du thread A\n");
    pthread_create(&thA, NULL, threadA, NULL);

    printf("Création du thread B\n");
    pthread_create(&thB, NULL, threadB, NULL);

    // attendre que les threads aient terminé
    pthread_join(thA, NULL);
    pthread_join(thB, NULL);

    printf("\nFin du thread principal\n");
    return 0;
}
