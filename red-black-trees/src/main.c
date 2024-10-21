#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "red_black_trees.h"

#define CREATE_NODE(x)                                                         \
  int *i##x = malloc(sizeof(int));                                             \
  *i##x = (x);                                                                 \
  node_t *n##x = malloc(sizeof(node_t));                                       \
  init_node(n##x, i##x);

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

void convert(char *buffer, void *data) { sprintf(buffer, "%d", *(int *)data); }

void print(node_t *node) { printf("%d\n", *(int *)node->data); }

void delete(node_t *node) {
  free(node->data);
  free(node);
}

int main(void) {
  {
    // stack example
    tree_t tree;
    init_tree(&tree, compare);

    int i0 = 0;
    node_t n0;
    init_node(&n0, &i0);
    rb_insert(&tree, &n0);

    int i1 = 1;
    node_t n1;
    init_node(&n1, &i1);
    rb_insert(&tree, &n1);

    int i2 = 2;
    node_t n2;
    init_node(&n2, &i2);
    rb_insert(&tree, &n2);

    int i3 = 3;
    node_t n3;
    init_node(&n3, &i3);
    rb_insert(&tree, &n3);

    node_t *node;
    rb_remove_key(&tree, &i0, &node);

    traverse_tree(&tree, print);
  }

  printf("\n\n");

  {
    // heap example
    tree_t *tree = malloc(sizeof(tree_t));
    init_tree(tree, compare);

    CREATE_NODE(0)
    rb_insert(tree, n0);

    CREATE_NODE(1)
    rb_insert(tree, n1);

    CREATE_NODE(2)
    rb_insert(tree, n2);

    CREATE_NODE(3)
    rb_insert(tree, n3);

    rb_print_tree(tree, stdout, convert, 8);

    traverse_tree(tree, delete);
    free(tree);
  }

  return 0;
}
