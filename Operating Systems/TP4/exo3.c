#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX 10

int A[MAX][MAX], B[MAX][MAX], C[MAX][MAX];
int n, m, p; // Dimensions

typedef struct {
    int row; // row number to compute
} Args;

void* multiply_row(void* arg) {
    int i = ((Args*)arg)->row;

    for (int j = 0; j < p; j++) {
        C[i][j] = 0;
        for (int k = 0; k < m; k++)
            C[i][j] += A[i][k] * B[k][j];
    }

    pthread_exit(NULL);
}

int main() {
    printf("Dimensions de A (n x m): ");
    scanf("%d %d", &n, &m);
    printf("Nombre de colonnes de B: ");
    scanf("%d", &p);

    printf("Matrice A:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &A[i][j]);

    printf("Matrice B:\n");
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            scanf("%d", &B[i][j]);

    pthread_t threads[MAX];
    Args args[MAX];

    for (int i = 0; i < n; i++) {
        args[i].row = i;
        pthread_create(&threads[i], NULL, multiply_row, &args[i]);
    }

    for (int i = 0; i < n; i++)
        pthread_join(threads[i], NULL);

    printf("Résultat C = A × B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }

    return 0;
}
