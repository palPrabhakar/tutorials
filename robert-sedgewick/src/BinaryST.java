import edu.princeton.cs.algs4.*;

public class BinaryST {
	
	private node root = null;
	
	private class node { 
		private int key;
		private int value;
		private node left = null;
		private node right = null;
		private int size;
		
		public node (int Key, int Val, int n) {
			this.key = Key;
			this.value = Val;
			this.size = n;
		}
	}
	
	public void put(int Key, int Val) {
		root = put(root, Key, Val);
	}
	
	private node put(node current, int Key, int Val) {
		if (current == null) {
			return current = new node(Key, Val, 1);
		}
		else if (less(current, Val)) {
			current.left = put(current.left, Key, Val);
		}
		else {
			current.right = put(current.right, Key, Val);
		}
		
		current.size = size(current.left) + size(current.right) + 1;
		return current;
		 
	}
	
	public void deleteMax() {
		root = deleteMax(root);
	}
	
	private node deleteMax(node current) {
		if (current == null) return null;
		else if (current.right == null) return current = current.left;
		else current.right = deleteMax(current.right);
		current.size = size(current.left) + size(current.right) + 1;
		return current;
	}
	
	public void deleteMin() {
		root = deleteMin(root);
	}
	
	private node deleteMin(node current) {
		if (current == null) return null;
		else if(current.left == null) return current.right; 
		else current.left = deleteMin(current.left);
		current.size = size(current.left) + size(current.right) + 1;
		return current;
	}

	public int max() {
		node x = max(root);
		if (x == null) return Integer.MIN_VALUE;
		return x.value;
	}
	
	private node max(node current) {
		if (current == null) return null;
		
		if(current.right == null) return current;
		else return max(current.right);
	}
	
	public int min() {
		node x = min(root);
		if (x == null) return Integer.MIN_VALUE;
		return x.value;
	}
	
	private node min(node current) {
		if (current == null) return null;
		
		if(current.left == null) return current;
		else return min(current.left);
	}
	
	public int floor(int key) {
		node x = floor(root, key);
		if(x == null) return Integer.MIN_VALUE;
		return x.key;
	}
	
	private node floor(node current, int key) {
		if(current == null) return current;
		
		if(key == current.key) return current;
		if(key < current.key) return floor(current.left, key);
		node t = floor(current.right, key);
		if(t != null) return t;
		else return current;
	}
	
	public int ceiling(int key) {
		node x = ceiling(root, key);
		if(x == null) return Integer.MIN_VALUE;
		else return x.key;
	}
	
	private node ceiling(node x, int key) {
		if (x == null) return x;
		
		if(key == x.key) return x;
		if(key > x.key) return ceiling(x.right, key);
		node t = ceiling(x.left, key);
		if (t != null) return t;
		else return x;
	}
	
	public int height() {
		return height(root);
	}
	
	private int height(node x) {
		if (x == null) return -1;
		return 1 + Math.max(height(x.right), height(x.left));
	}
	
	public boolean less(node x, int val) {
		return val < x.key;
	}
	
	public int size() {
		return size(root);
	}
	
	private int size(node current) {
		if (current == null) return 0;
		else return current.size;
	}
	
	public int get(int key) {
		return get(root, key);
	}
	
	private int get(node current, int key) {
		if (current == null) return Integer.MIN_VALUE;
		int val = current.value;
		
		if(current.key != key) {
			if (less(current, key)) val = get(current.left, key);
			else val = get(current.right, key);
		}
		
		return val;
	}
	
	public boolean isEmpty() {
		return root == null;
	}
	
	public void print() {
		print(root);
		System.out.printf("\n");
	}
	
	private void print(node current) {
		if(current == null) return;
		
		print(current.left);
		System.out.printf("%d ", current.key);
		print(current.right);
		
	}
	
	public boolean contains(int key) {
		return get(key)  != Integer.MIN_VALUE;
	}
	
	public static void main(String[] args) {
		int n = Integer.parseInt(args[0]);
		int key, val;
		BinaryST bst = new BinaryST();
		/*
		bst.put(5, 5);
		bst.put(6, 6);
		bst.put(2, 2);
		*/
		
		while (n > 0) {
			key = StdIn.readInt();
			val = key;
			bst.put(key, val);
			n--;
		}
		
		bst.print();
		//System.out.printf("\n %d \n",bst.value(5));
		//System.out.printf("%d \n",bst.value(11));
		//System.out.printf("%d \n",bst.value(2));
		//System.out.printf("%d \n",bst.value(7));
		System.out.printf("\n floor : %d \n",bst.ceiling(76));
		//bst.deleteMax();
		/*
		bst.deleteMax();
		System.out.printf("\n size : %d \n",bst.size());
		bst.print();
		bst.deleteMax();
		System.out.printf("\n size : %d \n",bst.size());
		bst.print();
		bst.deleteMax();
		System.out.printf("\n size : %d \n",bst.size());
		bst.print();
		bst.deleteMax();
		System.out.printf("\n size : %d \n",bst.size());
		bst.print();
		bst.deleteMax();
		System.out.printf("\n size : %d \n",bst.size());
		bst.print();
		bst.deleteMax();
		System.out.printf("\n size : %d \n",bst.size());
		bst.print();
		*/
		//System.out.printf("\n max : %d \n",bst.max());
		//System.out.printf("size : %d \n",bst.size());
		
	}
	

}
