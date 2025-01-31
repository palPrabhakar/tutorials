import java.io.Console;
import java.util.Queue;
import edu.princeton.cs.algs4.Bag;
import edu.princeton.cs.algs4.In;

public class EdgeWeighterGraph {
	private Bag<Edge>[] adj;
	private final int v;
	private int e; 
	
	public EdgeWeighterGraph(int v) {
		this.v = v;
		adj = (Bag<Edge>[]) new Bag[v];
		for (int i = 0; i < v; i++) {
			adj[i] = new Bag<Edge>();
		}
	}
	
	public EdgeWeighterGraph(In in) {
		this(in.readInt());
		int E = in.readInt();
		for (int i = 0; i < E; i++) {
			Edge e = new Edge(in.readInt(), in.readInt(), in.readDouble());
			addEdge(e);
		}
	}
	
	public void addEdge(Edge e) {
		adj[e.either()].add(e);
		adj[e.other(e.either())].add(e);
		this.e++;
	}
		
	public int v() {
		return this.v;
	}
	
	public int e() { 
		return this.e;
	}
	
	public Iterable<Edge> adj(int v) {
		return adj[v];
	}
	
	public Iterable<Edge> edges() {
		Bag<Edge> bag = new Bag<Edge>();
		for (int i = 0; i < v; i++){
			for(Edge w : adj[i]){
				bag.add(w);
			}
		}
		return bag;
	}
	
	public static void main(String[] args) {
		EdgeWeighterGraph g = new EdgeWeighterGraph(new In(args[0]));
		/*
		for(Edge e : g.adj(0)) {
			System.out.println(e);
		}
		*/
		//EagerPrimMST mst = new EagerPrimMST(g);
		KruskalMST mst = new KruskalMST(g);
		
		for(Edge e : mst.mst()){
			System.out.println(e);
		}
		
		System.out.println(mst.weight());
	}
}
