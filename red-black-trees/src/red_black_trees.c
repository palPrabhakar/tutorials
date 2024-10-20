#include "red_black_trees.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tree {
  node_t *root;
  comparator f;
  size_t n;
};

struct node {
  struct node *p, *l, *r;
  color_t color;
  void *data;
};

static node_t nil = {.color = black};

static void rb_insert_fixup(tree_t *tree, node_t *node);
static void rb_rotate_left(tree_t *tree, node_t *node);
static void rb_rotate_right(tree_t *tree, node_t *node);

tree_t *init_tree(comparator f) {
  tree_t *tree = malloc(sizeof(tree_t));
  tree->root = &nil;
  tree->f = f;
  tree->n = 0;
  return tree;
}

node_t *create_node(void *data) {
  node_t *node = malloc(sizeof(node_t));
  node->color = red;
  node->data = data;
  node->r = &nil;
  node->l = &nil;
  node->p = &nil;
  return node;
}

void rb_insert(tree_t *tree, node_t *node) {
  node_t *parent = &nil;
  node_t *child = tree->root;

  while (child != &nil) {
    parent = child;
    if (tree->f(node->data, child->data)) {
      child = child->l;
    } else {
      child = child->r;
    }
  }

  node->p = parent;
  if (parent == &nil) {
    tree->root = node;
  } else if (tree->f(node->data, parent->data)) {
    parent->l = node;
  } else {
    parent->r = node;
  }

  rb_insert_fixup(tree, node);
  tree->n++;
}

void rb_insert_fixup(tree_t *tree, node_t *node) {
  node_t *uncle;
  while (node->p->color == red) {
    if (node->p == node->p->p->l) {
      uncle = node->p->p->r;
      if (uncle->color == red) {
        node->p->color = black;
        uncle->color = black;
        node->p->p->color = red;
        node = node->p->p;
      } else {
        if (node == node->p->r) {
          node = node->p;
          rb_rotate_left(tree, node);
        }

        node->p->color = black;
        node->p->p->color = red;

        rb_rotate_right(tree, node->p->p);
      }
    } else {
      uncle = node->p->p->l;
      if (uncle->color == red) {
        node->p->color = black;
        uncle->color = black;
        node->p->p->color = red;
        node = node->p->p;
      } else {
        if (node == node->p->l) {
          node = node->p;
          rb_rotate_right(tree, node);
        }

        node->p->color = black;
        node->p->p->color = red;
        rb_rotate_left(tree, node->p->p);
      }
    }
  }

  tree->root->color = black;
}

void rb_rotate_left(tree_t *tree, node_t *node) {
  node_t *child = node->r;
  node->r = child->l;
  if (child->l != &nil) {
    child->l->p = node;
  }

  child->p = node->p;
  if (node->p == &nil) {
    tree->root = child;
  } else if (node == node->p->l) {
    node->p->l = child;
  } else {
    node->p->r = child;
  }

  child->l = node;
  node->p = child;
}

void rb_rotate_right(tree_t *tree, node_t *node) {
  node_t *child = node->l;
  node->l = child->r;
  if (child->r != &nil) {
    child->r->p = node;
  }

  child->p = node->p;
  if (node->p == &nil) {
    tree->root = child;
  } else if (node == node->p->l) {
    node->p->l = child;
  } else {
    node->p->r = child;
  }

  child->r = node;
  node->p = child;
}

static const char *node_color(color_t color) {
  switch (color) {
  case red:
    return "red";
  case black:
    return "black";
  default:
    return "";
  }
}

