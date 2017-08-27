import edu.princeton.cs.algs4.*;

public class DoublyLinkedList<Item> {
	
	private DoubleNode head;
	private DoubleNode last;
	private int N;
	
	private class DoubleNode<Item> {
		private Item item;
		private DoubleNode next;
		private DoubleNode prev;
	}
	
	public boolean isEmpty() {
		return N == 0;
	}
	
	public int size() {
		return N;
	}
	
	public void insertBegin(Item item) {
		DoubleNode newNode = new DoubleNode();
		newNode.item = item;
		newNode.next = head;
		if(isEmpty()){
			head = newNode;
			last = head;
			head.prev = null;
		}
		else{
			head.prev = newNode;
			head = newNode;
		}
	
		N++;
	}
	
	public void insertEnd(Item item) {
		DoubleNode newNode = new DoubleNode();
		newNode.item = item;
		if(isEmpty()) {
			last = newNode;
			last.next = null;
			last.prev = null;
			head = last;
		}
		else{
			last.next = newNode;
			newNode.prev = last;
			last = newNode;
		}
		N++;
	}
	
	public Item removeBegin() {
		if (isEmpty()) throw new RuntimeException("Queue underflow");
		Item item = (Item) head.item;
		head = head.next; 
		if(head != null) head.prev = null;
		N--;
		return item;
	}
	
	public Item removeEnd() {
		if (isEmpty()) throw new RuntimeException("Queue underflow");
		Item item = (Item) last.item;
		last = last.prev;
		if(last != null) last.next = null;
		N--;
		return item;
	}
	
	// Need to Improve this method 
	public void removeNode(int pos){
		if(pos <= 0 || pos > N) throw new RuntimeException("Index out of bounds");
		DoubleNode current;
		if(pos == 1){
			head = head.next;
			head.prev = null;
			N--;
		}
		else if(pos == N){
			last = last.prev;
			last.next = null;
			N--;
		}
		else{
			int count = 1;
			current = head;
			while(count != pos){
				current = current.next;
				count++;
			}
			current.next.prev = current.prev;
			current.prev.next = current.next;
			N--;
		}
		
	}
	
	public static void main(String[] args) {
		DoublyLinkedList<Integer> dl = new DoublyLinkedList<Integer>();
		
		dl.insertBegin(3);
		dl.insertBegin(2);
		dl.insertBegin(1);
		dl.insertEnd(4);
		dl.insertEnd(5);
		
		dl.removeNode(3);
		
		StdOut.println(dl.removeEnd());
		StdOut.println(dl.removeEnd());
		StdOut.println(dl.removeBegin());
		StdOut.println(dl.removeBegin());
		//StdOut.println(dl.removeBegin());
		
		StdOut.println(dl.size());
	}
	
}
