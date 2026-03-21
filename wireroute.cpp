/**
 * Parallel VLSI Wire Routing via MPI
 * Michael Nguyen(mguyen3), Ankita Kundu(akundu2)
 */

 #include "wireroute.h"

 #include <algorithm>
 #include <cassert>
 #include <chrono>
 #include <cmath>
 #include <cstring>
 #include <fstream>
 #include <iomanip>
 #include <iostream>
 #include <numeric>
 #include <random>
 #include <string>
 #include <vector>
 
 #include <mpi.h>
 #include <unistd.h>
 
 void print_stats(const std::vector<int> &occupancy) {
   int max_occupancy = 0;
   long long total_cost = 0;
   for (int val : occupancy) {
     max_occupancy = std::max(max_occupancy, val);
     total_cost += (long long)val * val;
   }
   std::cout << "Max occupancy: " << max_occupancy << '\n';
   std::cout << "Total cost: " << total_cost << '\n';
 }
 
 /* This function write the output into 2 files
 (1) It write occupancy grids into a file
 (2) It convert wires from Wire to validate_wire_t by to_validate_format
 (2) It write wires into another file
 */
 void write_output(
     const std::vector<Wire> &wires, const int num_wires,
     const std::vector<int> &occupancy, const int dim_x,
     const int dim_y,
     std::string wires_output_file_path = "outputs/wire_output.txt",
     std::string occupancy_output_file_path = "outputs/occ_output.txt") {
 
   std::ofstream out_occupancy(occupancy_output_file_path, std::fstream::out);
   if (!out_occupancy) {
     std::cerr << "Unable to open file: " << occupancy_output_file_path << '\n';
     exit(EXIT_FAILURE);
   }
   out_occupancy << dim_x << ' ' << dim_y << '\n';
 
   for (int y = 0; y < dim_y; y++) {
     for (int x = 0; x < dim_x; x++)
       out_occupancy << occupancy[y * dim_x + x] << (x == dim_x - 1 ? "" : " ");
     out_occupancy << '\n';
   }
   out_occupancy.close();
 
   std::ofstream out_wires(wires_output_file_path, std::fstream::out);
   if (!out_wires) {
     std::cerr << "Unable to open file: " << wires_output_file_path << '\n';
     exit(EXIT_FAILURE);
   }
 
   out_wires << dim_x << ' ' << dim_y << '\n';
   out_wires << num_wires << '\n';
 
   for (const auto &wire : wires) {
     // NOTICE: we convert to keypoint representation here, using
     // to_validate_format which need to be defined in the bottom of this file
     validate_wire_t keypoints = wire.to_validate_format();
     for (int i = 0; i < keypoints.num_pts; ++i) {
       out_wires << keypoints.p[i].x << ' ' << keypoints.p[i].y;
       if (i < keypoints.num_pts - 1)
         out_wires << ' ';
     }
     out_wires << '\n';
   }
 
   out_wires.close();
 }
 
 /* Helper function to get the key points of wire. Used by update_occupancy and current_route_cost*/
 inline int get_keypoints(const Wire &wire, int points_x[], int points_y[]) {
   int num_points = wire.num_bends + 2;
   for (int i = 0; i < num_points; i++) {
     if (i == 0) {
       points_x[i] = wire.start_x;
       points_y[i] = wire.start_y;
     } else if (i == num_points - 1) {
       points_x[i] = wire.end_x;
       points_y[i] = wire.end_y;
     } else {
       points_x[i] = wire.bend_x[i - 1];
       points_y[i] = wire.bend_y[i - 1];
     }
   }
   return num_points;
 }
 
 /* Helper to update the grid occupancy values by delta (+- 1) based on wire */
 inline void update_occupancy(std::vector<int> &occupancy, int dim_x, const Wire &wire, int delta) {
   int *__restrict__ occ = occupancy.data();
   int points_x[MAX_PTS_PER_WIRE];
   int points_y[MAX_PTS_PER_WIRE];
   int num_points = get_keypoints(wire, points_x, points_y);
   int num_segments = num_points - 1;
 
   for (int s = 0; s < num_segments; s++) {
     int cur_x = points_x[s];
     int cur_y = points_y[s];
     int next_x = points_x[s + 1];
     int next_y = points_y[s + 1];
 
     if (cur_y == next_y) {
       int *row = occ + cur_y * dim_x;
       int step = (cur_x < next_x) ? 1 : -1;
       for (int col = cur_x; col != next_x; col += step)
         row[col] += delta;
     } else {
       int idx = cur_y * dim_x + cur_x;
       int step = (cur_y < next_y) ? dim_x : -dim_x;
       int len = std::abs(next_y - cur_y);
       for (int i = 0; i < len; i++, idx += step)
         occ[idx] += delta;
     }
   }
   occ[points_y[num_points - 1] * dim_x + points_x[num_points - 1]] += delta;
 }
 
 /* Helper to calculate cost of wire's existing route as baseline against candidate costs */
 inline int current_route_cost(const std::vector<int> &occupancy, int dim_x, const Wire &wire) {
   const int *__restrict__ occ = occupancy.data();
   int cost = 0;
   int points_x[MAX_PTS_PER_WIRE];
   int points_y[MAX_PTS_PER_WIRE];
   int num_points = get_keypoints(wire, points_x, points_y);
   int num_segments = num_points - 1;
 
   for (int s = 0; s < num_segments; s++) {
     int cur_x = points_x[s];
     int cur_y = points_y[s];
     int next_x = points_x[s + 1];
     int next_y = points_y[s + 1];
 
     if (cur_y == next_y) {
       const int *row = occ + cur_y * dim_x;
       int step = (cur_x < next_x) ? 1 : -1;
       for (int col = cur_x; col != next_x; col += step) {
         int v = row[col];
         cost += (v + 1) * (v + 1);
       }
     } else {
       int idx = cur_y * dim_x + cur_x;
       int step = (cur_y < next_y) ? dim_x : -dim_x;
       int len = std::abs(next_y - cur_y);
       for (int i = 0; i < len; i++, idx += step) {
         int v = occ[idx];
         cost += (v + 1) * (v + 1);
       }
     }
   }
   int v = occ[points_y[num_points - 1] * dim_x + points_x[num_points - 1]];
   cost += (v + 1) * (v + 1);
   return cost;
 }
 
 /* Helper to calculate cost of horizontal segment of wire */
 static inline int h_seg_cost(const int *__restrict__ occ, int dim_x, int y, int from_x, int to_x) {
   int cost = 0;
   const int *row = occ + y * dim_x;
   int step = (from_x < to_x) ? 1 : -1;
   for (int col = from_x; col != to_x; col += step) {
     int v = row[col];
     cost += (v + 1) * (v + 1);
   }
   return cost;
 }
 
 /* Helper to calculate cost of vertical segment of wire */
 static inline int v_seg_cost(const int *__restrict__ occ, int dim_x, int x, int from_y, int to_y) {
   int cost = 0;
   int idx = from_y * dim_x + x;
   int step = (from_y < to_y) ? dim_x : -dim_x;
   int len = std::abs(to_y - from_y);
   for (int i = 0; i < len; i++, idx += step) {
     int v = occ[idx];
     cost += (v + 1) * (v + 1);
   }
   return cost;
 }
 
 /*
  * Cost of an L-shaped path
  * horizontal from (x0, y0) to (x1, y0),
  * then vertical from (x1, y0) to (x1, y1)
  */
 static inline int l_path_cost(const int *__restrict__ occ, int dim_x,
                                int x0, int y0, int x1, int y1, bool horizontal_first) {
   if (horizontal_first) {
     return h_seg_cost(occ, dim_x, y0, x0, x1) + v_seg_cost(occ, dim_x, x1, y0, y1);
   } else {
     return v_seg_cost(occ, dim_x, x0, y0, y1) + h_seg_cost(occ, dim_x, y1, x0, x1);
   }
 }
 
 /* Score a candidate route without constructing the wire */
 static inline int candidate_route_cost(const std::vector<int> &occupancy, int dim_x,
                                        int start_x, int start_y, int end_x, int end_y,
                                        int route_index) {
   const int *__restrict__ occ = occupancy.data();
 
   int dx = std::abs(end_x - start_x);
   int dy = std::abs(end_y - start_y);
   int dir_x = (end_x > start_x) ? 1 : -1;
   int dir_y = (end_y > start_y) ? 1 : -1;
   int min_x = std::min(start_x, end_x);
   int min_y = std::min(start_y, end_y);
 
   int cost = 0;
 
   if (route_index < dx) {
     // horizontal-first 1 or 2 bend
     int offset = route_index + 1;
     if (offset == dx) {
       cost += l_path_cost(occ, dim_x, start_x, start_y, end_x, end_y, true);
     } else {
       int split_x = start_x + dir_x * offset;
       cost += l_path_cost(occ, dim_x, start_x, start_y, split_x, end_y, true);
       cost += h_seg_cost(occ, dim_x, end_y, split_x, end_x);
     }
   } else if (route_index < dx + dy) {
     // vertical-first 1 or 2 bend
     int offset = route_index - dx + 1;
     if (offset == dy) {
       cost += l_path_cost(occ, dim_x, start_x, start_y, end_x, end_y, false);
     } else {
       int split_y = start_y + dir_y * offset;
       cost += l_path_cost(occ, dim_x, start_x, start_y, end_x, split_y, false);
       cost += v_seg_cost(occ, dim_x, end_x, split_y, end_y);
     }
   } else {
     // 3-bend through interior point (mid_x, mid_y)
     int flat_idx = route_index - dx - dy;
     int horizontal_first = flat_idx % 2;
     int midpoint_idx = flat_idx / 2;
     int inner_cols = dx - 1;
     int mid_x = min_x + (midpoint_idx % inner_cols) + 1;
     int mid_y = min_y + (midpoint_idx / inner_cols) + 1;
     if (horizontal_first == 0) {
       cost += l_path_cost(occ, dim_x, start_x, start_y, mid_x, mid_y, true);
       cost += l_path_cost(occ, dim_x, mid_x, mid_y, end_x, end_y, true);
     } else {
       cost += l_path_cost(occ, dim_x, start_x, start_y, mid_x, mid_y, false);
       cost += l_path_cost(occ, dim_x, mid_x, mid_y, end_x, end_y, false);
     }
   }
   // Add endpoint too
   int endpoint_v = occ[end_y * dim_x + end_x];
   cost += (endpoint_v + 1) * (endpoint_v + 1);
   return cost;
 }
 
 /* Construct wire struct given a route index */
 inline Wire construct_route(const Wire &wire, int route_index) {
   Wire result;
   int start_x = wire.start_x, start_y = wire.start_y;
   int end_x = wire.end_x, end_y = wire.end_y;
   int dx = std::abs(end_x - start_x);
   int dy = std::abs(end_y - start_y);
   int dir_x = (end_x > start_x) ? 1 : -1;
   int dir_y = (end_y > start_y) ? 1 : -1;
 
   // endpoints don't change 
   result.start_x = start_x;
   result.start_y = start_y;
   result.end_x = end_x;
   result.end_y = end_y;
 
   if (route_index < dx) {
     // horizontal-first 1 or 2 bend
     int offset = route_index + 1;
     if (offset == dx) {
       result.num_bends = 1;
       result.bend_x[0] = end_x;
       result.bend_y[0] = start_y;
     } else {
       int split_x = start_x + dir_x * offset;
       result.num_bends = 2;
       result.bend_x[0] = split_x;
       result.bend_y[0] = start_y;
       result.bend_x[1] = split_x;
       result.bend_y[1] = end_y;
     }
   } else if (route_index < dx + dy) {
     // vertical-first 1 or 2 bend
     int offset = route_index - dx + 1;
     if (offset == dy) {
       result.num_bends = 1;
       result.bend_x[0] = start_x;
       result.bend_y[0] = end_y;
     } else {
       int split_y = start_y + dir_y * offset;
       result.num_bends = 2;
       result.bend_x[0] = start_x;
       result.bend_y[0] = split_y;
       result.bend_x[1] = end_x;
       result.bend_y[1] = split_y;
     }
   } else {
     // 3-bend through interior point (mid_x, mid_y)
     int flat_idx = route_index - dx - dy;
     int point_idx = flat_idx / 2;
     int horizontal_first = flat_idx % 2;
     int interior_cols = dx - 1;
     int min_x = std::min(start_x, end_x);
     int min_y = std::min(start_y, end_y);
     int mid_x = min_x + (point_idx % interior_cols) + 1;
     int mid_y = min_y + (point_idx / interior_cols) + 1;
     if (horizontal_first == 0) {
       // Horizontal -> Vertical -> Horizontal -> Vertical
       result.bend_x[0] = mid_x; 
       result.bend_y[0] = start_y;
       result.bend_x[1] = mid_x; 
       result.bend_y[1] = mid_y;
       result.bend_x[2] = end_x; 
       result.bend_y[2] = mid_y;
     } else {
       // Vertical -> Horizontal -> Vertical -> Horizontal
       result.bend_x[0] = start_x;  
       result.bend_y[0] = mid_y;
       result.bend_x[1] = mid_x;
       result.bend_y[1] = mid_y; 
       result.bend_x[2] = mid_x;
       result.bend_y[2] = end_y;  
     }
     result.num_bends = 3;
   }
   return result;
 }
 
 /* Pack a Wire into a RouteUpdate message for MPI communication */
 RouteUpdate pack_update(int wire_id, const Wire &w) {
   RouteUpdate msg;
   msg.num_bends = w.num_bends;
   msg.wire_id = wire_id;
   msg.start_x = w.start_x;
   msg.start_y = w.start_y;
   for (int i = 0; i < 3; i++) {
     msg.bend_x[i] = w.bend_x[i];
     msg.bend_y[i] = w.bend_y[i];
   }
   msg.end_x = w.end_x;
   msg.end_y = w.end_y;
   
   return msg;
 }
 
 /* Unpack a RouteUpdate message back into a Wire */
 Wire unpack_update(const RouteUpdate &msg) {
   Wire w;
   w.num_bends = msg.num_bends;
   w.start_x = msg.start_x;
   w.start_y = msg.start_y;
   for (int i = 0; i < 3; i++) {
     w.bend_x[i] = msg.bend_x[i];
     w.bend_y[i] = msg.bend_y[i];
   }
   w.end_x = msg.end_x;
   w.end_y = msg.end_y;
 
   return w;
 }
 
 /* Apply a batch of route updates to the occupancy grid and wire list */
 void commit_updates(std::vector<int> &occupancy, int dim_x, std::vector<Wire> &wires,
                     const std::vector<RouteUpdate> &updates) {
   for (const RouteUpdate &msg : updates) {
     update_occupancy(occupancy, dim_x, wires[msg.wire_id], -1);
     wires[msg.wire_id] = unpack_update(msg);
     update_occupancy(occupancy, dim_x, wires[msg.wire_id], +1);
   }
 }
 
 /* Serialize route updates into a byte buffer: [src_rank, count, updates...] */
 int serialize_updates(char *buf, int src_rank, const std::vector<RouteUpdate> &updates) {
   int count = static_cast<int>(updates.size());
   std::memcpy(buf, &src_rank, sizeof(int));
   std::memcpy(buf + sizeof(int), &count, sizeof(int));
   if (count > 0)
     std::memcpy(buf + 2 * sizeof(int), updates.data(), count * sizeof(RouteUpdate));
   return static_cast<int>(2 * sizeof(int) + count * sizeof(RouteUpdate));
 }
 
 /* Deserialize route updates from a byte buffer */
 void deserialize_updates(const char *buf, int &src_rank, std::vector<RouteUpdate> &updates) {
   int count = 0;
   std::memcpy(&src_rank, buf, sizeof(int));
   std::memcpy(&count, buf + sizeof(int), sizeof(int));
   updates.resize(count);
   if (count > 0)
     std::memcpy(updates.data(), buf + 2 * sizeof(int), count * sizeof(RouteUpdate));
 }
 
 /* Ring-based allgather of route updates across all MPI ranks */
 void ring_exchange(std::vector<int> &occupancy, int dim_x, std::vector<Wire> &wires,
                    const std::vector<RouteUpdate> &local_updates, int capacity, 
                    int rank, int nproc, int tag,
                    std::vector<char> &send_buf,
                    std::vector<char> &recv_buf) {
   const int max_bytes = static_cast<int>(2 * sizeof(int) + capacity * sizeof(RouteUpdate));
   const int left = (rank - 1 + nproc) % nproc;
   const int right = (rank + 1) % nproc;
 
   std::vector<RouteUpdate> outgoing = local_updates;
   std::vector<RouteUpdate> incoming;
   int src_rank = rank;
   int recv_rank = -1;
 
   for (int step = 0; step < nproc - 1; step++) {
     int send_size = serialize_updates(send_buf.data(), src_rank, outgoing);
 
     MPI_Request reqs[2];
     MPI_Irecv(recv_buf.data(), max_bytes, MPI_BYTE, left, tag,
               MPI_COMM_WORLD, &reqs[0]);
     MPI_Isend(send_buf.data(), send_size, MPI_BYTE, right, tag,
               MPI_COMM_WORLD, &reqs[1]);
 
     if (src_rank != rank)
       commit_updates(occupancy, dim_x, wires, outgoing);
 
     MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
     deserialize_updates(recv_buf.data(), recv_rank, incoming);
 
     outgoing.swap(incoming);
     src_rank = recv_rank;
   }
 
   if (src_rank != rank)
     commit_updates(occupancy, dim_x, wires, outgoing);
 }
 
 /* Number of candidate routes for a wire with given dx, dy */
 inline long long count_routes(int dx, int dy) {
   if (dx == 0 || dy == 0) return 0;
   return (long long)dx + dy + 2LL * (dx - 1) * (dy - 1);
 }
 
 int main(int argc, char *argv[]) {
   const auto init_start = std::chrono::steady_clock::now();
   int rank = 0;
   int nproc = 0;
 
   // Initialize MPI
   MPI_Init(&argc, &argv);
   // Get process rank
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   // Get total number of processes  
   MPI_Comm_size(MPI_COMM_WORLD, &nproc);
 
   std::string input_filename;
   double SA_prob = 0.1;
   int SA_iters = 5;
   int batch_size = 1;
  int n_flag = 0;
  char parallel_mode = '\0';
  bool profile_hotloops = false;
 
   // Read command line arguments
   int opt;
  while ((opt = getopt(argc, argv, "f:n:p:i:m:b:P")) != -1) {
     switch (opt) {
       case 'f':
         input_filename = optarg;
         break;
      case 'n':
        n_flag = atoi(optarg);
        break;
       case 'p':
         SA_prob = atof(optarg);
         break;
       case 'i':
         SA_iters = atoi(optarg);
         break;
      case 'm':
        parallel_mode = optarg[0];
        break;
       case 'b':
         batch_size = atoi(optarg);
         break;
      case 'P':
        profile_hotloops = true;
        break;
       default:
         if (rank == 0) {
          std::cerr << "Usage: " << argv[0]
                    << " -f input_filename -n num_procs [-p SA_prob] "
                       "[-i SA_iters] -m parallel_mode -b batch_size [-P]\n";
         }
 
         MPI_Finalize();
         exit(EXIT_FAILURE);
     }
   }
 
   // Check if required options are provided
  if (input_filename.empty() || n_flag <= 0 || SA_iters <= 0 ||
      batch_size <= 0 || (parallel_mode != 'A' && parallel_mode != 'W')) {
     if (rank == 0) {
      std::cerr << "Usage: " << argv[0]
                << " -f input_filename -n num_procs [-p SA_prob] "
                   "[-i SA_iters] -m parallel_mode -b batch_size [-P]\n";
     }
 
     MPI_Finalize();
     exit(EXIT_FAILURE);
   }

  if (n_flag != nproc) {
    if (rank == 0) {
      std::cerr << "Error: -n " << n_flag
                << " must match number of MPI processes " << nproc << "\n";
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (parallel_mode != 'A') {
    if (rank == 0) {
      std::cerr << "This implementation supports only "
                   "across-wires mode (-m A).\n";
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
 
   if (rank == 0) {
     std::cout << "Number of processes: " << nproc << '\n';
     std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
     std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
     std::cout << "Input file: " << input_filename << '\n';
    std::cout << "Parallel mode: " << parallel_mode << '\n';
     std::cout << "Batch size: " << batch_size << '\n';
    if (profile_hotloops) {
      std::cout << "Profiling: enabled (-P)\n";
    }
   }
 
   int dim_x, dim_y, num_wires;
   std::vector<Wire> wires;
   std::vector<int> wire_order;
 
   // each rank reads input individually 
   std::ifstream fin(input_filename);
   if (!fin) {
     std::cerr << "Rank " << rank << ": unable to open " << input_filename
               << '\n';
     MPI_Finalize();
     exit(EXIT_FAILURE);
   }
   fin >> dim_x >> dim_y >> num_wires;
 
   wires.resize(num_wires);
   wire_order.resize(num_wires);
   std::iota(wire_order.begin(), wire_order.end(), 0);
   for (auto &wire : wires) {
     fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
     if (wire.start_x != wire.end_x && wire.start_y != wire.end_y) {
       wire.num_bends = 1;
       wire.bend_x[0] = wire.end_x;
       wire.bend_y[0] = wire.start_y;
     } else {
       wire.num_bends = 0;
     }
   }
   std::vector<int> occupancy(dim_y * dim_x, 0);
   for (const auto &wire : wires)
     update_occupancy(occupancy, dim_x, wire, +1);
   // Sort wires by number of candidate routes (descending) for load balancing 
   std::stable_sort(wire_order.begin(), wire_order.end(),
     [&](int a, int b) {
       long long ra = count_routes(std::abs(wires[a].end_x - wires[a].start_x),
                                   std::abs(wires[a].end_y - wires[a].start_y));
       long long rb = count_routes(std::abs(wires[b].end_x - wires[b].start_x),
                                   std::abs(wires[b].end_y - wires[b].start_y));
       if (ra == rb) {
         return a < b; 
       } else {
         return ra > rb;
       }
     }
   );
 
   if (rank == 0) {
     const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
     std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';
   }
 
   const auto compute_start = std::chrono::steady_clock::now();
 
   // Core routing and communication logic 
   std::mt19937 rng(rank + 42);
   std::uniform_real_distribution<double> probability(0.0, 1.0);
 
   const int wires_per_round = (nproc == 1) ? num_wires : batch_size;
   const int msg_capacity = std::max(1, wires_per_round);
   const size_t buf_bytes = 2 * sizeof(int) + msg_capacity * sizeof(RouteUpdate);
   std::vector<char> send_buf;
   std::vector<char> recv_buf;
   if (nproc > 1) {
     send_buf.resize(buf_bytes);
     recv_buf.resize(buf_bytes);
   }
 
   const int num_batches = (num_wires + (wires_per_round * nproc) - 1) / (wires_per_round * nproc);
 
   for (int iter = 0; iter < SA_iters; iter++) {
     for (int batch = 0; batch < num_batches; batch++) {
       std::vector<RouteUpdate> local_updates;
 
       const int round_start = batch * wires_per_round;
       const int round_end = round_start + wires_per_round;
       for (int idx = round_start; idx < round_end; idx++) {
         int global_idx = rank + idx * nproc;
         if (global_idx >= num_wires) {
           break;
         }
         int wire_id = wire_order[global_idx];
 
         Wire &wire = wires[wire_id];
         int dx = std::abs(wire.end_x - wire.start_x);
         int dy = std::abs(wire.end_y - wire.start_y);
         long long num_routes = count_routes(dx, dy);
         if (num_routes == 0) continue;
 
         if (probability(rng) < SA_prob) {
           std::uniform_int_distribution<int> route_dist(0, static_cast<int>(num_routes) - 1);
           Wire new_wire = construct_route(wire, route_dist(rng));
           update_occupancy(occupancy, dim_x, wire, -1);
           wire = new_wire;
           update_occupancy(occupancy, dim_x, wire, +1);
           local_updates.push_back(pack_update(wire_id, wire));
           continue;
         }
 
         update_occupancy(occupancy, dim_x, wire, -1);
 
         int lowest_cost = current_route_cost(occupancy, dim_x, wire);
         int best_route = -1;
 
         for (int r = 0; r < static_cast<int>(num_routes); r++) {
           int c = candidate_route_cost(occupancy, dim_x,
                                        wire.start_x, wire.start_y,
                                        wire.end_x, wire.end_y, r);
           if (c < lowest_cost) {
             lowest_cost = c;
             best_route = r;
           }
         }
 
         if (best_route >= 0) {
           wire = construct_route(wire, best_route);
           local_updates.push_back(pack_update(wire_id, wire));
         }
         update_occupancy(occupancy, dim_x, wire, +1);
       }
 
       if (nproc > 1) {
         const int tag = (iter * num_batches + batch) % 12345;
         ring_exchange(occupancy, dim_x, wires, local_updates, msg_capacity,
                       rank, nproc, tag, send_buf, recv_buf);
       }
     }
   }
 
   MPI_Barrier(MPI_COMM_WORLD);
 
   if (rank == 0) {
     const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
     std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
   }
 
   if (rank == 0) {
     std::vector<std::vector<int>> occupancy_2d(dim_y, std::vector<int>(dim_x));
     for (int y = 0; y < dim_y; y++)
       for (int x = 0; x < dim_x; x++)
         occupancy_2d[y][x] = occupancy[y * dim_x + x];
     wr_checker checker(wires, occupancy_2d);
     checker.validate();
     print_stats(occupancy);
     write_output(wires, num_wires, occupancy, dim_x, dim_y);
   }
 
   // Cleanup
   MPI_Finalize();
   return 0;
 }
 
 
 /* TODO (student): implement to_validate_format to convert Wire to
   validate_wire_t keypoint representation in order to run checker and
   write output
 */
 validate_wire_t Wire::to_validate_format(void) const {
   validate_wire_t w;
   w.num_pts = static_cast<uint8_t>(2 + num_bends);
   for (int i = 0; i < w.num_pts; i++) {
     if (i == 0) {
       w.p[i].x = static_cast<uint16_t>(start_x);
       w.p[i].y = static_cast<uint16_t>(start_y);
     } else if (i == w.num_pts - 1) {
       w.p[i].x = static_cast<uint16_t>(end_x);
       w.p[i].y = static_cast<uint16_t>(end_y);
     } else {
       w.p[i].x = static_cast<uint16_t>(bend_x[i-1]);
       w.p[i].y = static_cast<uint16_t>(bend_y[i-1]);
     }
   }
   
   return w;
 }
 