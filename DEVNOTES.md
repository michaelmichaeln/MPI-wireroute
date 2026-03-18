LAB 4 NOTES

V1: private copies of shared space, all to all communication, non-blocking send rcv
* computes wires, waits for all messages to be sent and recieved to everyone in their private buffer space :MPI_waitall
* every process updates their private space with the cumulative buffer changes
* Bad: lots of memory usage really bad for bg boards, frequent message passing. n/b rounds with p*(p-1) messages each round. dominated by communication time.

- exploding memory p x board memory

V2: Use a unidirectional forwarding ring with single-bucket messages and shift-level overlap.
1. The Setup (curr_bucket = my_updates)
You start the phase. Your local grid is already updated with the wires you just routed. You package those exact updates into curr_bucket to send to your neighbor.
2. The Asynchronous Kickoff (MPI_Isend / MPI_Irecv)
You tell MPI, "Start receiving a bucket from my left neighbor into next_bucket, and start sending curr_bucket to my right neighbor." Because you use Isend and Irecv (the 'I' stands for immediate/non-blocking), your code does not pause here to wait for the network. It immediately moves to the next line.
3. The "Overlap" (The Performance Secret)
if shift > 0: apply curr_bucket to local occupancy/wires
This is the magic step. While the network hardware is busy moving data between nodes in the background, your CPU doesn't sit idle. Instead, it processes the data it received in the previous shift. You are essentially hiding the time it takes to apply the updates behind the time it takes to send the messages.
4. The Synchronization (MPI_Waitall)
Now your CPU has finished applying the updates. It hits MPI_Waitall. If the network is still sending/receiving, the CPU pauses here. If the network is already done, it passes right through. This ensures you don't overwrite buffers before the network is finished with them.
5. The Swap (swap curr_bucket <-> next_bucket)
The next_bucket you just received from your left neighbor becomes your curr_bucket. In the next loop iteration, you will pass this newly acquired bucket to your right neighbor.
6. Final Cleanup
After $P-1$ iterations, the loop ends. However, you still have the very last bucket you received sitting in your hands. You apply it to your local grid to complete the sync.

Findings: ring with single-bucket messages and shift-level overlap
improvesments: O(B) message sizes, reusing buckets lets us have O(P*B) memory usage, no dynamic allocations, still communicating too often, BUS BOTTLENECK

Does not scale with p , or b both cosntant


V3:
changes: capped U routes 80k -> 256
performance:
similar cost
still not good but seeing speedup, 3.4x benchtime for p=1
to much busy time

V4: Capping l and z routes aswell

 OK idk why but the baseline computation is just really high im restarting :P, evaluation was wrong a
FInal iter: Bucket passing, 
* hides latency by using non-blocking MPI_Ircv/send, send your accumulated batch changes away to the next rank through your bucket  to the right and recieve another bucket from the left. -> can work on your bucket while sending yours away
* Once the network confirms the new bucket from your left neighbor has fully arrived, you unpack it, swap your buffers, and loop backevery rank has a private occupancy grid, 
* all-to-all style route exchanges using exchange_updates (performs better with large b as comm/work is increaed). 
* interleaved batch assignment
* for non straight wire
* TLDR: everyone computes their batch changes, then sends it around to one neighbor. there are p rounds of batch exhcanges where every processor records the batch changes.
    * larger b -> staler memory but less communication
exchange_updates(...) does symmetric ring propagation:
* Pack [origin_rank, count, RouteUpdate[]] into send_buf.
    * For nproc - 1 shifts:
    * MPI_Irecv from prev, MPI_Isend to next
    * while transfer in flight, apply current remote bucket (if not own)
    * wait, unpack next bucket


mnguyen3@ghc53:~/private/15418/asst4/code$ ./run_wires_bench.sh -s f -m g -p 0.1 -b 32 -r 3 -n 1,2,4,8
=== wireroute benchmark ===
  wire set:    f (./inputs/timeinput/few_wires.txt)
  machine:    g
  output CSV: ./bench_results.csv
  SA prob:    0.1
  SA iters:   5
  batch size: 32
  repeats:    3
  procs:      1 2 4 8

p    compute_median     speedup    efficiency   cost_median 
--------------------------------------------------------------------------
1    3.0813189940       1.0000     1.0000       29242       
2    1.6746298030       1.8400     0.9200       29242       
4    0.9325697010       3.3041     0.8260       29238       
8    0.5807391490       5.3059     0.6632       29250 
