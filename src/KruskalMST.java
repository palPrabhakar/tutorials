import edu.princeton.cs.algs4.*;

public class KruskalMST {
	private Queue<Edge> mst;
	private MinPQ<Edge> pq;
	private UF uf;
	private double weight;
	
	public KruskalMST(EdgeWeighterGraph g) {
		mst = new Queue<Edge>();
		uf = new UF(g.v());
		pq = new MinPQ<Edge>();
		
		for(int i = 0; i < g.v(); i++) {
			for(Edge e : g.adj(i)) {
				pq.insert(e);
			}
		}
		
		while(mst.size() != g.v() - 1) {
			Edge e = pq.delMin();
			if(!uf.connected(e.either(), e.other(e.either()))) {
				uf.union(e.either(), e.other(e.either()));
				weight += e.weight();
				mst.enqueue(e);
			}	
		}
	}
	
	public Iterable<Edge> mst() {
		return mst;
	}
	
	public double weight() {
		return weight;
	}
	
}
