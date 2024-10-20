#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "red_black_trees.h"
#include "queue.h"

int compare(void *a, void *b) {
  int *x = a;
  int *y = b;
  return *x < *y;
}

int main(void) {
  tree_t *rbt = init_tree(compare);

  int *iptr = malloc(sizeof(int));
  *iptr = 1;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 2;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 3;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 4;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 5;
  rb_insert(rbt, create_node(iptr));

  // iptr = malloc(sizeof(int));
  // *iptr = 6;
  // rb_insert(rbt, create_node(iptr));
  //
  // iptr = malloc(sizeof(int));
  // *iptr = 7;
  // rb_insert(rbt, create_node(iptr));
  //
  // iptr = malloc(sizeof(int));
  // *iptr = 8;
  // rb_insert(rbt, create_node(iptr));
  //
  // iptr = malloc(sizeof(int));
  // *iptr = 9;
  // rb_insert(rbt, create_node(iptr));

  // rb_print_tree(rbt);

  return 0;
}
