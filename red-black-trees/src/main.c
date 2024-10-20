#include "red_black_trees.h"
#include <stdio.h>
#include <stdlib.h>

int compare(void *a, void *b) {
  int *x = a;
  int *y = b;
  return *x < *y;
}

int main(void) {
  printf("Red Black Trees\n");

  tree_t *rbt = init_tree(compare);
  
  int *iptr = malloc(sizeof(int));
  *iptr = 1;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 2;
  rb_insert(rbt, create_node(iptr));

  return 0;
}
