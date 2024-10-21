#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "red_black_trees.h"

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

  iptr = malloc(sizeof(int));
  *iptr = 6;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 7;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 8;
  rb_insert(rbt, create_node(iptr));

  iptr = malloc(sizeof(int));
  *iptr = 9;
  rb_insert(rbt, create_node(iptr));

  // rb_delete(rbt, rb_root(rbt));
  int key = 5;
  rb_delete_key(rbt, &key);

  rb_print_tree(rbt, stdout, convert, 8);

  return 0;
}
