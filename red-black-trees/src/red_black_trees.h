#ifndef RBT_H
#define RBT_H
#include <stdio.h>

// return a < b
typedef int (*comparator)(void *, void *);

typedef enum color { red, black } color_t;

typedef struct node {
  struct node *p, *l, *r;
  color_t color;
  void *data;
} node_t;

typedef struct tree {
  node_t *root;
  comparator f;
  size_t n;
} tree_t;

// void *: pointer to data elem
// returns: red-black-tree node ready for insertion
//    -color: red
//    -p, l, r = &nil
node_t *create_node(void *);

// comparator: function used to compare keys
// returns: a red-black-tree 
tree_t *init_tree(comparator);

// tree_t *: valid tree pointer
//    - use init_tree to create a valid tree_t*
// node_t *: valid node_t *
//    - valid node_t *:
//      - data: key which can be compared using the comparator
//      - color: red
//      - p, l, r = &nil
//    - use create_node to create new node
// rb_insert will not fail if above conditions are met
void rb_insert(tree_t *, node_t *);

// TODO: 
// int rb_delete_key(tree_t *tree, void *key)
//    - top level fn which can be called to delete a specific key from the tree
//    - returns:
//        - 0: if key exists in the tree (successful delete!)
//        - 1: if key doesn't exists in the tree

// tree_t *: valid tree pointer 
// node_t *: a node which exist in the tree
// rb_delete will not fail if above conditions are met
void rb_delete(tree_t *, node_t *);

// FILE *: write to file
// to_string: function to convert data to char *
//    -char *: buffer
//    -void *: data
// size_t: buffer_size
// buffer_size should be large enough to
// hold char * converted data
typedef void (*to_string)(char *, void *);
void rb_print_tree(tree_t *, FILE *, to_string, size_t);

// tree_t *: a valid tree pointer 
// returns:
//    - tree root node
node_t *rb_root(tree_t *);

#endif
