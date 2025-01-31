import edu.princeton.cs.algs4.*;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Queue_29<Item> implements Iterable<Item>{
	
	@SuppressWarnings("rawtypes")
	private Node last;
	private int N;
	
	private class Node<Item> {
		Item item;
		Node next; 
	}
	
	public Queue_29(){
		last = null;
	}
	
	public boolean isEmpty(){
		return N == 0;
	}
	
	public int size(){
		return N;
	}
	
	public void enqueue(Item item) {
		Node newNode = new Node();
		if(last == null){
			newNode.item = item;
			last = newNode;
			last.next = last;
		}
		else{
			newNode.item = item;
			newNode.next = last.next;
			last.next = newNode;
			last = newNode;
		}
		N++;
	}
	
	public Item dequeue() {
		if (isEmpty()) throw new RuntimeException("Queue underflow");
		Item item = (Item) last.next.item;
		if(last.next == last) last = null;
		last.next = last.next.next;
		N--;
		return item;
	}
	
	public Iterator<Item> iterator(){
		return new ListIterator();
	}
	
	public String toString(){
		StringBuilder s = new StringBuilder();
        for (Item item : this)
            s.append(item + " ");
        return s.toString();
	}
	
	private class ListIterator implements Iterator<Item> {
		private Node current = last;
		private int n = N;
		
		public boolean hasNext(){
			return n > 0;
		}
		
		public void remove() { }
		
		public Item next() {
			Item item = (Item) current.next.item;
			current = current.next;
			n--;
			return item;
		}
	}
	
	public static void main(String[] args) {
        Queue_29<String> q = new Queue_29<String>();
        while (!StdIn.isEmpty()) {
            String item = StdIn.readString();
            if (!item.equals("-")) q.enqueue(item);
            else if (!q.isEmpty()) StdOut.print(q.dequeue() + " ");
        }
        StdOut.println("(" + q.size() + " left on queue: [ " + q + "])");
        
        for(String str : q){
        	StdOut.println(str);
        }
    }
	
}