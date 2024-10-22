#include <stdio.h>
#include <string.h>

#include "../src/red_black_trees.h"

FILE *tmp;

#define T_INSERT(x, t, exp)                                                    \
  int i##x = x;                                                                \
  node_t n##x;                                                                 \
  init_node(&n##x, &i##x);                                                     \
  rb_insert(t, &n##x);                                                         \
  fseek(tmp, 0, SEEK_SET);                                                     \
  traverse_tree(t, print);                                                     \
  {                                                                            \
    char buffer[2 * t->size];                                                  \
    fseek(tmp, 0, SEEK_SET);                                                   \
    fgets(buffer, 2 * t->size + 1, tmp);                                       \
    if (strncmp(buffer, exp, 2 * tree->size) != 0) {                           \
      printf("Error Input (%s) != Expected (%s).\n", buffer, exp);             \
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

// Should do a BFS traversal for better results
void test_insertion(tree_t *tree) {
  T_INSERT(0, tree, "0b");

  T_INSERT(1, tree, "0b1r");

  T_INSERT(2, tree, "0r1b2r");

  T_INSERT(3, tree, "0b1b2b3r");
}

int main() {
  printf("Testing red-black-tress.\n");

  tmp = tmpfile();

  tree_t tree;
  init_tree(&tree, compare);

  test_insertion(&tree);

  fclose(tmp);

  printf("Finished testing red-black-trees.\n");

  return 0;
}
