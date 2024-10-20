#include <stdio.h>
#include <stdlib.h>

#include "../../queue/src/queue.h"
#include "red_black_trees.h"

#define CHECK_QUEUE(func, ...)                                                 \
  if (func) {                                                                  \
    fprintf(stderr, __VA_ARGS__);                                              \
    break;                                                                     \
  }

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

static void rb_insert_fixup(tree_t *, node_t *);
static void rb_rotate_left(tree_t *, node_t *);
static void rb_rotate_right(tree_t *, node_t *);
static void rb_transplant(tree_t *, node_t *, node_t *);
static void rb_delete_fixup(tree_t *, node_t *);
static node_t *tree_minimum(node_t *);

node_t *tree_minimum(node_t *node) {
  while (node->l != &nil) {
    node = node->l;
  }

  return node;
}

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

node_t *rb_root(tree_t *tree) {
  return tree->root;
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

void rb_transplant(tree_t *tree, node_t *u, node_t *v) {
  if (u->p == &nil) {
    tree->root = v;
  } else if (u == u->p->l) {
    u->p->l = v;
  } else {
    u->p->r = v;
  }
  v->p = u->p;
}

void rb_delete(tree_t *tree, node_t *node) {
  node_t *tmp = node;
  color_t tmp_color = tmp->color;
  node_t *tmp_succ;

  if (node->l == &nil) {
    tmp_succ = node->r;
    rb_transplant(tree, node, node->r);
  } else if (node->r == &nil) {
    tmp_succ = node->l;
    rb_transplant(tree, node, node->l);
  } else {
    tmp = tree_minimum(node->r);
    tmp_color = tmp->color;
    tmp_succ = tmp->r;

    if (tmp->p == node) {
      /*
      Although you might think that setting x.p to y in line 16 is unnecessary
      since x is a child of y, the call of RB-DELETE-FIXUP relies on x.p being
      y even if x is T.nil. -Cormen et. al 4th ed.
      */
      tmp_succ->p = tmp;
    } else {
      rb_transplant(tree, tmp, tmp->r);
      tmp->r = node->r;
      tmp->r->p = tmp;
    }

    rb_transplant(tree, node, tmp);
    tmp->l = node->l;
    tmp->l->p = tmp;
    tmp->color = node->color;
  }

  if (tmp_color == black) {
    rb_delete_fixup(tree, tmp_succ);
  }
}

void rb_delete_fixup(tree_t *tree, node_t *node) {
  // If node is a red-black node then skip while loop and
  // simply color the node black
  // If node is root then skip while loop and simply
  // remove the blackness
  // else:
  node_t *sibling;
  while (node != tree->root && node->color == black) {
    if (node == node->p->l) {
      sibling = node->p->r;

      if (sibling->color == red) {
        sibling->color = black;
        node->p->color = red;
        rb_rotate_left(tree, node->p);
        sibling = node->p->r;
      }

      if (sibling->l->color == black && sibling->r->color) {
        sibling->color = red;
        node = node->p;
      } else {
        if (sibling->r->color == black) {
          sibling->l->color = black;
          sibling->color = red;
          rb_rotate_right(tree, sibling);
          sibling = node->p->r;
        }

        sibling->color = node->p->color;
        node->p->color = black;
        sibling->r->color = black;
        rb_rotate_left(tree, node->p);
        node = tree->root;
      }
    } else {
      sibling = node->p->l;

      if (sibling->color == red) {
        sibling->color = black;
        node->p->color = red;
        rb_rotate_left(tree, node->p);
        sibling = node->p->l;
      }

      if (sibling->l->color == black && sibling->r->color == black) {
        sibling->color = red;
        node = node->p;
      } else {
        if(sibling->l->color == black) {
          sibling->r->color = black;
          sibling->color = red;
          rb_rotate_left(tree, sibling);
          sibling = node->p->l;
        }

        sibling->color = node->p->color;
        node->p->color = black;
        sibling->l->color = black;
        rb_rotate_right(tree, node->p);
        node = tree->root;
      }
    }
  }

  node->color = black;
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

void rb_print_tree(tree_t *tree, FILE *f) {
  fprintf(f, "digraph RedBlackTree {\n");
  fprintf(f, "node [shape=circle style=filled];\n");

  if (tree->root != &nil) {

    queue_t *queue = init_queue(tree->n);

    queue_push(queue, tree->root);

    while (queue_size(queue) != 0) {
      node_t *node;
      CHECK_QUEUE(queue_pop(queue, (void **)&node),
                  "Error: queue_pop() failed at %d.\n", __LINE__)

      fprintf(f, "%d [fillcolor=%s fontcolor=white];\n", *(int *)node->data,
              node_color(node->color));

      if (node->l != &nil) {
        fprintf(f, "%d -> %d;\n", *(int *)node->data, *(int *)node->l->data);
        CHECK_QUEUE(queue_push(queue, node->l),
                    "Error: queue_push() failed at %d.\n", __LINE__);
      }

      if (node->r != &nil) {
        fprintf(f, "%d -> %d;\n", *(int *)node->data, *(int *)node->r->data);
        CHECK_QUEUE(queue_push(queue, node->r),
                    "Error: queue_push() failed at %d.\n", __LINE__);
      }
    }
  }

  fprintf(f, "}\n");
}
