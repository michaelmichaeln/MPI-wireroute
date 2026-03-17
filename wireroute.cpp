/**
 * Parallel VLSI Wire Routing via MPI
 * Assignment 4: Pure across-wires, symmetric peer-to-peer.
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

void print_stats(const int *occ, int n) {
  int max_occupancy = 0;
  long long total_cost = 0;
  for (int i = 0; i < n; i++) {
    max_occupancy = std::max(max_occupancy, occ[i]);
    total_cost += (long long)occ[i] * occ[i];
  }
  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

void write_output(
    const std::vector<Wire> &wires, const int num_wires,
    const int *occ, const int dim_x, const int dim_y,
    std::string wires_output_file_path,
    std::string occupancy_output_file_path) {

  std::ofstream out_occupancy(occupancy_output_file_path, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }
  out_occupancy << dim_x << ' ' << dim_y << '\n';
  for (int y = 0; y < dim_y; y++) {
    for (int x = 0; x < dim_x; x++) {
      out_occupancy << occ[y * dim_x + x];
      if (x < dim_x - 1) out_occupancy << ' ';
    }
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
    validate_wire_t keypoints = wire.to_validate_format();
    for (int i = 0; i < keypoints.num_pts; ++i) {
      out_wires << keypoints.p[i].x << ' ' << keypoints.p[i].y;
      if (i < keypoints.num_pts - 1) out_wires << ' ';
    }
    out_wires << '\n';
  }
  out_wires.close();
}

inline void update_occupancy(int *occ, int dim_x,
                             const Wire &wire, int delta) {
  int num_pts = 2 + wire.num_bends;
  int px[MAX_PTS_PER_WIRE], py[MAX_PTS_PER_WIRE];
  px[0] = wire.start_x; py[0] = wire.start_y;
  for (int i = 0; i < wire.num_bends; i++) {
    px[1 + i] = wire.bend_x[i];
    py[1 + i] = wire.bend_y[i];
  }
  px[num_pts - 1] = wire.end_x;
  py[num_pts - 1] = wire.end_y;

  for (int seg = 0; seg < num_pts - 1; seg++) {
    int x = px[seg], y = py[seg];
    int x_end = px[seg + 1], y_end = py[seg + 1];
    if (y == y_end) {
      int *row = occ + y * dim_x;
      if (x < x_end) {
        for (int xi = x; xi < x_end; xi++) row[xi] += delta;
      } else {
        for (int xi = x; xi > x_end; xi--) row[xi] += delta;
      }
    } else {
      if (y < y_end) {
        for (int yi = y, idx = y * dim_x + x; yi < y_end; yi++, idx += dim_x)
          occ[idx] += delta;
      } else {
        for (int yi = y, idx = y * dim_x + x; yi > y_end; yi--, idx -= dim_x)
          occ[idx] += delta;
      }
    }
  }
  occ[py[num_pts - 1] * dim_x + px[num_pts - 1]] += delta;
}

inline int path_cost_direct(const int *occ, int dim_x, const Wire &wire) {
  int cost = 0;
  int num_pts = 2 + wire.num_bends;
  int px[MAX_PTS_PER_WIRE], py[MAX_PTS_PER_WIRE];
  px[0] = wire.start_x; py[0] = wire.start_y;
  for (int i = 0; i < wire.num_bends; i++) {
    px[1 + i] = wire.bend_x[i];
    py[1 + i] = wire.bend_y[i];
  }
  px[num_pts - 1] = wire.end_x;
  py[num_pts - 1] = wire.end_y;

  for (int seg = 0; seg < num_pts - 1; seg++) {
    int x = px[seg], y = py[seg];
    int x_end = px[seg + 1], y_end = py[seg + 1];
    if (y == y_end) {
      const int *row = occ + y * dim_x;
      if (x < x_end) {
        for (int xi = x; xi < x_end; xi++) {
          int v = row[xi];
          cost += (v + 1) * (v + 1);
        }
      } else {
        for (int xi = x; xi > x_end; xi--) {
          int v = row[xi];
          cost += (v + 1) * (v + 1);
        }
      }
    } else {
      if (y < y_end) {
        for (int yi = y, idx = y * dim_x + x; yi < y_end; yi++, idx += dim_x) {
          int v = occ[idx];
          cost += (v + 1) * (v + 1);
        }
      } else {
        for (int yi = y, idx = y * dim_x + x; yi > y_end; yi--, idx -= dim_x) {
          int v = occ[idx];
          cost += (v + 1) * (v + 1);
        }
      }
    }
  }
  int v = occ[py[num_pts - 1] * dim_x + px[num_pts - 1]];
  cost += (v + 1) * (v + 1);
  return cost;
}

static inline int horizontal_seg_cost(
    const int *__restrict__ occ, int dim_x, int y, int x_from, int x_to) {
  int cost = 0;
  const int *row = occ + y * dim_x;
  if (x_from < x_to) {
    for (int x = x_from; x < x_to; x++) {
      int v = row[x];
      cost += (v + 1) * (v + 1);
    }
  } else {
    for (int x = x_from; x > x_to; x--) {
      int v = row[x];
      cost += (v + 1) * (v + 1);
    }
  }
  return cost;
}

static inline int vertical_seg_cost(
    const int *__restrict__ occ, int dim_x, int x, int y_from, int y_to) {
  int cost = 0;
  if (y_from < y_to) {
    for (int y = y_from, idx = y_from * dim_x + x; y < y_to; y++, idx += dim_x) {
      int v = occ[idx];
      cost += (v + 1) * (v + 1);
    }
  } else {
    for (int y = y_from, idx = y_from * dim_x + x; y > y_to; y--, idx -= dim_x) {
      int v = occ[idx];
      cost += (v + 1) * (v + 1);
    }
  }
  return cost;
}

static inline int eval_route_cost(
    const int *__restrict__ occ, int dim_x,
    int sx, int sy, int ex, int ey,
    int abs_dx, int abs_dy, int sign_x, int sign_y,
    int min_x, int min_y,
    int route_index) {

  int cost = 0;

  if (route_index < abs_dx) {
    int a = route_index + 1;
    int bx = sx + sign_x * a;
    if (a == abs_dx) {
      cost += horizontal_seg_cost(occ, dim_x, sy, sx, ex);
      cost += vertical_seg_cost(occ, dim_x, ex, sy, ey);
    } else {
      cost += horizontal_seg_cost(occ, dim_x, sy, sx, bx);
      cost += vertical_seg_cost(occ, dim_x, bx, sy, ey);
      cost += horizontal_seg_cost(occ, dim_x, ey, bx, ex);
    }
  } else if (route_index < abs_dx + abs_dy) {
    int b = route_index - abs_dx + 1;
    int by = sy + sign_y * b;
    if (b == abs_dy) {
      cost += vertical_seg_cost(occ, dim_x, sx, sy, ey);
      cost += horizontal_seg_cost(occ, dim_x, ey, sx, ex);
    } else {
      cost += vertical_seg_cost(occ, dim_x, sx, sy, by);
      cost += horizontal_seg_cost(occ, dim_x, by, sx, ex);
      cost += vertical_seg_cost(occ, dim_x, ex, by, ey);
    }
  } else {
    int r_rel = route_index - abs_dx - abs_dy;
    int interior_idx = r_rel / 2;
    int orientation = r_rel % 2;
    int interior_w = abs_dx - 1;
    int xp = min_x + 1 + (interior_idx % interior_w);
    int yp = min_y + 1 + (interior_idx / interior_w);
    if (orientation == 0) {
      cost += horizontal_seg_cost(occ, dim_x, sy, sx, xp);
      cost += vertical_seg_cost(occ, dim_x, xp, sy, yp);
      cost += horizontal_seg_cost(occ, dim_x, yp, xp, ex);
      cost += vertical_seg_cost(occ, dim_x, ex, yp, ey);
    } else {
      cost += vertical_seg_cost(occ, dim_x, sx, sy, yp);
      cost += horizontal_seg_cost(occ, dim_x, yp, sx, xp);
      cost += vertical_seg_cost(occ, dim_x, xp, yp, ey);
      cost += horizontal_seg_cost(occ, dim_x, ey, xp, ex);
    }
  }

  int v = occ[ey * dim_x + ex];
  cost += (v + 1) * (v + 1);
  return cost;
}

struct PerfStats {
  double t_occ = 0.0;
  double t_eval = 0.0;
  double t_search = 0.0;
  long long occ_calls = 0;
  long long route_evals = 0;
};

inline Wire build_route(const Wire &wire, int route_index) {
  int sx = wire.start_x, sy = wire.start_y;
  int ex = wire.end_x, ey = wire.end_y;
  int abs_dx = std::abs(ex - sx);
  int abs_dy = std::abs(ey - sy);
  int sign_x = (ex > sx) ? 1 : -1;
  int sign_y = (ey > sy) ? 1 : -1;

  Wire w;
  w.start_x = sx;
  w.start_y = sy;
  w.end_x = ex;
  w.end_y = ey;

  if (route_index < abs_dx) {
    int a = route_index + 1;
    int bx = sx + sign_x * a;
    if (a == abs_dx) {
      w.num_bends = 1;
      w.bend_x[0] = ex;
      w.bend_y[0] = sy;
    } else {
      w.num_bends = 2;
      w.bend_x[0] = bx;
      w.bend_y[0] = sy;
      w.bend_x[1] = bx;
      w.bend_y[1] = ey;
    }
  } else if (route_index < abs_dx + abs_dy) {
    int b = route_index - abs_dx + 1;
    int by = sy + sign_y * b;
    if (b == abs_dy) {
      w.num_bends = 1;
      w.bend_x[0] = sx;
      w.bend_y[0] = ey;
    } else {
      w.num_bends = 2;
      w.bend_x[0] = sx;
      w.bend_y[0] = by;
      w.bend_x[1] = ex;
      w.bend_y[1] = by;
    }
  } else {
    int r_rel = route_index - abs_dx - abs_dy;
    int interior_idx = r_rel / 2;
    int orientation = r_rel % 2;
    int xp = std::min(sx, ex) + 1 + (interior_idx % (abs_dx - 1));
    int yp = std::min(sy, ey) + 1 + (interior_idx / (abs_dx - 1));
    w.num_bends = 3;
    if (orientation == 0) {
      w.bend_x[0] = xp;  w.bend_y[0] = sy;
      w.bend_x[1] = xp;  w.bend_y[1] = yp;
      w.bend_x[2] = ex;  w.bend_y[2] = yp;
    } else {
      w.bend_x[0] = sx;  w.bend_y[0] = yp;
      w.bend_x[1] = xp;  w.bend_y[1] = yp;
      w.bend_x[2] = xp;  w.bend_y[2] = ey;
    }
  }
  return w;
}

RouteUpdate wire_to_update(int wire_id, const Wire &w) {
  RouteUpdate u;
  u.wire_id = wire_id;
  u.start_x = w.start_x;
  u.start_y = w.start_y;
  u.end_x = w.end_x;
  u.end_y = w.end_y;
  u.num_bends = w.num_bends;
  for (int i = 0; i < 3; i++) {
    u.bend_x[i] = w.bend_x[i];
    u.bend_y[i] = w.bend_y[i];
  }
  return u;
}

Wire update_to_wire(const RouteUpdate &u) {
  Wire w;
  w.start_x = u.start_x;
  w.start_y = u.start_y;
  w.end_x = u.end_x;
  w.end_y = u.end_y;
  w.num_bends = u.num_bends;
  for (int i = 0; i < 3; i++) {
    w.bend_x[i] = u.bend_x[i];
    w.bend_y[i] = u.bend_y[i];
  }
  return w;
}

void apply_route_updates(int *occ, int dim_x,
                         std::vector<Wire> &wires,
                         const std::vector<RouteUpdate> &updates) {
  for (const RouteUpdate &u : updates) {
    int wi = u.wire_id;
    if (wi < 0 || wi >= static_cast<int>(wires.size())) continue;
    update_occupancy(occ, dim_x, wires[wi], -1);
    wires[wi] = update_to_wire(u);
    update_occupancy(occ, dim_x, wires[wi], +1);
  }
}

void exchange_updates(int *occ, int dim_x,
                      std::vector<Wire> &wires,
                      const std::vector<RouteUpdate> &my_updates,
                      int max_count, int rank, int nproc, int tag,
                      std::vector<char> &send_buf,
                      std::vector<char> &recv_buf) {
  const int max_bytes = static_cast<int>(2 * sizeof(int)) +
                        max_count * static_cast<int>(sizeof(RouteUpdate));

  auto pack_bucket = [&](int origin_rank,
                         const std::vector<RouteUpdate> &updates) -> int {
    int count = static_cast<int>(updates.size());
    if (count > max_count) {
      std::cerr << "Rank " << rank << ": too many route updates ("
                << count << " > " << max_count << ")\n";
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::memcpy(send_buf.data(), &origin_rank, sizeof(int));
    std::memcpy(send_buf.data() + sizeof(int), &count, sizeof(int));
    if (count > 0)
      std::memcpy(send_buf.data() + 2 * sizeof(int), updates.data(),
                  count * sizeof(RouteUpdate));
    return static_cast<int>(2 * sizeof(int)) +
           count * static_cast<int>(sizeof(RouteUpdate));
  };

  auto unpack_bucket = [&](std::vector<RouteUpdate> &updates,
                           int &origin_rank) {
    int count = 0;
    std::memcpy(&origin_rank, recv_buf.data(), sizeof(int));
    std::memcpy(&count, recv_buf.data() + sizeof(int), sizeof(int));
    if (count < 0 || count > max_count) {
      std::cerr << "Rank " << rank << ": invalid received count " << count
                << '\n';
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    updates.resize(count);
    if (count > 0)
      std::memcpy(updates.data(), recv_buf.data() + 2 * sizeof(int),
                  count * sizeof(RouteUpdate));
  };

  const int prev = (rank - 1 + nproc) % nproc;
  const int next = (rank + 1) % nproc;

  std::vector<RouteUpdate> curr_updates = my_updates;
  std::vector<RouteUpdate> next_updates;
  int curr_origin = rank;
  int next_origin = -1;

  for (int shift = 0; shift < nproc - 1; shift++) {
    int send_size = pack_bucket(curr_origin, curr_updates);

    MPI_Request reqs[2];
    MPI_Irecv(recv_buf.data(), max_bytes, MPI_BYTE, prev, tag,
              MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(send_buf.data(), send_size, MPI_BYTE, next, tag,
              MPI_COMM_WORLD, &reqs[1]);

    if (curr_origin != rank)
      apply_route_updates(occ, dim_x, wires, curr_updates);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    unpack_bucket(next_updates, next_origin);

    curr_updates.swap(next_updates);
    curr_origin = next_origin;
  }

  if (curr_origin != rank)
    apply_route_updates(occ, dim_x, wires, curr_updates);
}

int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();
  int rank = 0;
  int nproc = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::string input_filename;
  int n_flag = 0;
  double SA_prob = 0.1;
  int SA_iters = 5;
  char parallel_mode = '\0';
  int batch_size = 1;
  bool profile_hotloops = false;

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

  if (input_filename.empty() || n_flag <= 0 || SA_iters <= 0 ||
      batch_size <= 0 ||
      (parallel_mode != 'A' && parallel_mode != 'W')) {
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
    std::cout << "Simulated annealing probability parameter: " << SA_prob
              << '\n';
    std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
    std::cout << "Input file: " << input_filename << '\n';
    std::cout << "Parallel mode: " << parallel_mode << '\n';
    std::cout << "Batch size: " << batch_size << '\n';
  }

  int dim_x, dim_y, num_wires;
  std::vector<Wire> wires;
  std::vector<int> work_order;

  {
    std::ifstream fin(input_filename);
    if (!fin) {
      std::cerr << "Rank " << rank << ": unable to open " << input_filename
                << '\n';
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
    fin >> dim_x >> dim_y >> num_wires;
    wires.resize(num_wires);
    work_order.resize(num_wires);
    std::iota(work_order.begin(), work_order.end(), 0);
    for (auto &wire : wires) {
      fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
      if (wire.start_x == wire.end_x || wire.start_y == wire.end_y) {
        wire.num_bends = 0;
      } else {
        wire.num_bends = 1;
        wire.bend_x[0] = wire.end_x;
        wire.bend_y[0] = wire.start_y;
      }
    }

    std::stable_sort(
        work_order.begin(), work_order.end(),
        [&](int a, int b) {
          const Wire &wa = wires[a];
          const Wire &wb = wires[b];
          int adx = std::abs(wa.end_x - wa.start_x);
          int ady = std::abs(wa.end_y - wa.start_y);
          int bdx = std::abs(wb.end_x - wb.start_x);
          int bdy = std::abs(wb.end_y - wb.start_y);
          long long ac = (adx == 0 || ady == 0)
                             ? 0LL
                             : (long long)adx + ady +
                                   2LL * (adx - 1) * (ady - 1);
          long long bc = (bdx == 0 || bdy == 0)
                             ? 0LL
                             : (long long)bdx + bdy +
                                   2LL * (bdx - 1) * (bdy - 1);
          if (ac != bc) return ac > bc;
          return a < b;
        });
  }

  std::vector<int> occupancy(dim_y * dim_x, 0);
  int *occ = occupancy.data();

  for (const auto &wire : wires)
    update_occupancy(occ, dim_x, wire, +1);

  if (rank == 0) {
    const double init_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - init_start)
            .count();
    std::cout << "Initialization time (sec): " << std::fixed
              << std::setprecision(10) << init_time << '\n';
  }

  const auto compute_start = std::chrono::steady_clock::now();

  std::mt19937 rng(42 + rank);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  PerfStats perf;

  const int effective_batch = (nproc == 1) ? num_wires : batch_size;
  const int max_updates_per_peer = std::max(1, effective_batch);
  const size_t max_exchange_bytes =
      2 * sizeof(int) + max_updates_per_peer * sizeof(RouteUpdate);
  std::vector<char> ring_send_buf;
  std::vector<char> ring_recv_buf;
  if (nproc > 1) {
    ring_send_buf.resize(max_exchange_bytes);
    ring_recv_buf.resize(max_exchange_bytes);
  }

  const int num_batches =
      (num_wires + (effective_batch * nproc) - 1) / (effective_batch * nproc);

  for (int iter = 0; iter < SA_iters; iter++) {
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
      std::vector<RouteUpdate> my_updates;

      const int local_start = batch_idx * effective_batch;
      const int local_end = local_start + effective_batch;
      for (int local_i = local_start; local_i < local_end; local_i++) {
        int work_slot = rank + local_i * nproc;
        if (work_slot >= num_wires) break;
        int wi = work_order[work_slot];

        Wire &wire = wires[wi];
        int abs_dx = std::abs(wire.end_x - wire.start_x);
        int abs_dy = std::abs(wire.end_y - wire.start_y);
        if (abs_dx == 0 || abs_dy == 0) continue;

        int total_routes = abs_dx + abs_dy + 2 * (abs_dx - 1) * (abs_dy - 1);

        if (prob_dist(rng) < SA_prob) {
          std::uniform_int_distribution<int> route_dist(0, total_routes - 1);
          Wire new_wire = build_route(wire, route_dist(rng));
          auto occ_t0 = std::chrono::steady_clock::now();
          update_occupancy(occ, dim_x, wire, -1);
          wire = new_wire;
          update_occupancy(occ, dim_x, wire, +1);
          if (profile_hotloops) {
            perf.t_occ += std::chrono::duration_cast<std::chrono::duration<double>>(
                              std::chrono::steady_clock::now() - occ_t0)
                              .count();
            perf.occ_calls += 2;
          }
          my_updates.push_back(wire_to_update(wi, wire));
          continue;
        }

        int sx = wire.start_x, sy = wire.start_y;
        int ex_ = wire.end_x, ey_ = wire.end_y;
        int sign_x = (ex_ > sx) ? 1 : -1;
        int sign_y = (ey_ > sy) ? 1 : -1;
        int min_x = std::min(sx, ex_);
        int min_y = std::min(sy, ey_);

        auto occ_t0 = std::chrono::steady_clock::now();
        update_occupancy(occ, dim_x, wire, -1);
        if (profile_hotloops) {
          perf.t_occ += std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::steady_clock::now() - occ_t0)
                            .count();
          perf.occ_calls++;
        }
        auto search_t0 = std::chrono::steady_clock::now();
        int best_cost = path_cost_direct(occ, dim_x, wire);
        auto eval_t0 = std::chrono::steady_clock::now();
        int best_idx = -1;

        for (int r = 0; r < total_routes; r++) {
          int c = eval_route_cost(occ, dim_x, sx, sy, ex_, ey_,
                                  abs_dx, abs_dy, sign_x, sign_y,
                                  min_x, min_y, r);
          if (c < best_cost) {
            best_cost = c;
            best_idx = r;
          }
        }
        if (profile_hotloops) {
          perf.t_eval += std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::steady_clock::now() - eval_t0)
                             .count();
          perf.route_evals += total_routes;
          perf.t_search += std::chrono::duration_cast<std::chrono::duration<double>>(
                               std::chrono::steady_clock::now() - search_t0)
                               .count();
        }

        if (best_idx >= 0) {
          wire = build_route(wire, best_idx);
          my_updates.push_back(wire_to_update(wi, wire));
        }
        occ_t0 = std::chrono::steady_clock::now();
        update_occupancy(occ, dim_x, wire, +1);
        if (profile_hotloops) {
          perf.t_occ += std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::steady_clock::now() - occ_t0)
                            .count();
          perf.occ_calls++;
        }
      }

      if (nproc > 1) {
        const int tag = (iter * num_batches + batch_idx) % 32768;
        exchange_updates(occ, dim_x, wires, my_updates, max_updates_per_peer,
                         rank, nproc, tag, ring_send_buf, ring_recv_buf);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    const double compute_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - compute_start)
            .count();
    std::cout << "Computation time (sec): " << std::fixed
              << std::setprecision(10) << compute_time << '\n';
  }

  if (profile_hotloops) {
    double local_times[3] = {perf.t_occ, perf.t_eval, perf.t_search};
    double global_times[3] = {0.0, 0.0, 0.0};
    long long local_counts[2] = {perf.occ_calls, perf.route_evals};
    long long global_counts[2] = {0, 0};

    MPI_Reduce(local_times, global_times, 3, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(local_counts, global_counts, 2, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
      double avg_occ = global_times[0] / nproc;
      double avg_eval = global_times[1] / nproc;
      double avg_search = global_times[2] / nproc;
      std::cout << "Profile avg occupancy time (sec): " << std::fixed
                << std::setprecision(10) << avg_occ << '\n';
      std::cout << "Profile avg route-eval time (sec): " << std::fixed
                << std::setprecision(10) << avg_eval << '\n';
      std::cout << "Profile avg search time (sec): " << std::fixed
                << std::setprecision(10) << avg_search << '\n';
      std::cout << "Profile occupancy calls (sum): " << global_counts[0] << '\n';
      std::cout << "Profile route evals (sum): " << global_counts[1] << '\n';
    }
  }

  if (rank == 0) {
    std::vector<std::vector<int>> occ_2d(dim_y, std::vector<int>(dim_x));
    for (int y = 0; y < dim_y; y++)
      for (int x = 0; x < dim_x; x++)
        occ_2d[y][x] = occ[y * dim_x + x];

    wr_checker checker(wires, occ_2d);
    checker.validate();
    print_stats(occ, dim_x * dim_y);
    write_output(wires, num_wires, occ, dim_x, dim_y,
                 "outputs/wire_output.txt", "outputs/occ_output.txt");
  }

  MPI_Finalize();
  return 0;
}

validate_wire_t Wire::to_validate_format(void) const {
  validate_wire_t w;
  w.num_pts = static_cast<uint8_t>(2 + num_bends);
  w.p[0].x = static_cast<uint16_t>(start_x);
  w.p[0].y = static_cast<uint16_t>(start_y);
  for (int i = 0; i < num_bends; i++) {
    w.p[1 + i].x = static_cast<uint16_t>(bend_x[i]);
    w.p[1 + i].y = static_cast<uint16_t>(bend_y[i]);
  }
  w.p[w.num_pts - 1].x = static_cast<uint16_t>(end_x);
  w.p[w.num_pts - 1].y = static_cast<uint16_t>(end_y);
  return w;
}
