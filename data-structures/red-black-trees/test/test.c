#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/red_black_trees.h"

FILE *tmp;

#define T_INSERT(x, t, sz, exp)                                                \
  int *i##x = malloc(sizeof(int));                                             \
  *i##x = x;                                                                   \
  node_t *n##x = malloc(sizeof(node_t));                                       \
  init_node(n##x, i##x);                                                       \
  rb_insert(t, n##x);                                                          \
  if (t->size != sz)                                                           \
    printf("Error tree->size (%zu) != expected size (%zu) at line %d.\n",      \
           t->size, sz, __LINE__);                                             \
  fseek(tmp, 0, SEEK_SET);                                                     \
  traverse_tree_bfs(t, print);                                                 \
  {                                                                            \
    char buffer[2 * t->size];                                                  \
    fseek(tmp, 0, SEEK_SET);                                                   \
    fgets(buffer, 2 * t->size + 1, tmp);                                       \
    if (strncmp(buffer, exp, 2 * tree->size) != 0) {                           \
      printf("Error Input (%s) != Expected (%s) at line %d.\n", buffer, exp,   \
             __LINE__);                                                        \
    }                                                                          \
  }

#define T_DELETE(x, t, sz, exp)                                                \
  node_t *n##x;                                                                \
  int i##x = x;                                                                \
  if (rb_remove_key(t, &i##x, &n##x))                                          \
    printf("Error deleting key %d.\n", x);                                     \
  free((n##x)->data);                                                          \
  free(n##x);                                                                  \
  if (t->size != sz)                                                           \
    printf("Error tree->size (%zu) != expected size (%zu) at line %d.\n",      \
           t->size, sz, __LINE__);                                             \
  fseek(tmp, 0, SEEK_SET);                                                     \
  traverse_tree_bfs(t, print);                                                 \
  {                                                                            \
    char buffer[2 * t->size];                                                  \
    fseek(tmp, 0, SEEK_SET);                                                   \
    fgets(buffer, 2 * t->size + 1, tmp);                                       \
    if (strncmp(buffer, exp, 2 * tree->size) != 0) {                           \
      printf("Error Input (%s) != Expected (%s) at line %d.\n", buffer, exp,   \
             __LINE__);                                                        \
    }                                                                          \
  }

int compare(void *a, void *b) {
  int *x = a;
  int *y = b;
  if (*x == *y) {
    return 0;
  } else if (*x < *y) {
    return -1;
  } else {
    return 1;
  }
}

char color_c[] = {'r', 'b'};
char values_c[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

void print(node_t *node) {
  fprintf(tmp, "%c%c", values_c[*(int *)node->data], color_c[node->color]);
}

void test_insertion(tree_t *tree) {
  T_INSERT(0, tree, 1LU, "0b");

  T_INSERT(1, tree, 2LU, "0b1r");

  T_INSERT(2, tree, 3LU, "1b0r2r");

  T_INSERT(3, tree, 4LU, "1b0b2b3r");

  T_INSERT(4, tree, 5LU, "1b0b3b2r4r");
}

void test_deletion(tree_t *tree) {
  T_DELETE(3, tree, 4LU, "1b0b4b2r");

  T_DELETE(1, tree, 3LU, "2b0b4b");

  T_DELETE(0, tree, 2LU, "2b4r");

  T_DELETE(2, tree, 1LU, "4b");

  T_DELETE(4, tree, 0LU, "");
}

int main() {
  printf("Testing red-black-tress.\n");

  tmp = tmpfile();

  tree_t tree;
  init_tree(&tree, compare);

  test_insertion(&tree);

  test_deletion(&tree);

  fclose(tmp);

  printf("Finished testing red-black-trees.\n");

  return 0;
}
