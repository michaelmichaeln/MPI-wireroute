/**
 * Parallel VLSI Wire Routing via MPI
 * Assignment 4: Pure across-wires, symmetric peer-to-peer. hi buddy
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
#include <random>
#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>

void print_stats(const std::vector<std::vector<int>> &occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto &row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

void write_output(
    const std::vector<Wire> &wires, const int num_wires,
    const std::vector<std::vector<int>> &occupancy, const int dim_x,
    const int dim_y,
    std::string wires_output_file_path,
    std::string occupancy_output_file_path) {

  std::ofstream out_occupancy(occupancy_output_file_path, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }
  out_occupancy << dim_x << ' ' << dim_y << '\n';

  for (const auto &row : occupancy) {
    for (size_t i = 0; i < row.size(); ++i)
      out_occupancy << row[i] << (i == row.size() - 1 ? "" : " ");
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
      if (i < keypoints.num_pts - 1)
        out_wires << ' ';
    }
    out_wires << '\n';
  }

  out_wires.close();
}

// --- Occupancy and cost (no atomics; private memory per rank) ---
void update_occupancy(std::vector<std::vector<int>> &occupancy,
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
    int dx = (x_end > x) ? 1 : (x_end < x) ? -1 : 0;
    int dy = (y_end > y) ? 1 : (y_end < y) ? -1 : 0;
    while (x != x_end || y != y_end) {
      occupancy[y][x] += delta;
      x += dx; y += dy;
    }
    if (seg == num_pts - 2) {
      occupancy[y][x] += delta;
    }
  }
}

int path_cost_direct(const std::vector<std::vector<int>> &occupancy,
                     const Wire &wire) {
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
    int dx = (x_end > x) ? 1 : (x_end < x) ? -1 : 0;
    int dy = (y_end > y) ? 1 : (y_end < y) ? -1 : 0;
    while (x != x_end || y != y_end) {
      int v = occupancy[y][x];
      cost += v * v;
      x += dx; y += dy;
    }
    if (seg == num_pts - 2) {
      cost += occupancy[y][x] * occupancy[y][x];
    }
  }
  return cost;
}

void mask_wire_cells(const Wire &wire, std::vector<uint8_t> &mask, int dim_x,
                     uint8_t val) {
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
    int dx = (x_end > x) ? 1 : (x_end < x) ? -1 : 0;
    int dy = (y_end > y) ? 1 : (y_end < y) ? -1 : 0;
    while (x != x_end || y != y_end) {
      mask[y * dim_x + x] = val;
      x += dx; y += dy;
    }
    if (seg == num_pts - 2) {
      mask[y * dim_x + x] = val;
    }
  }
}

int path_cost_masked(const std::vector<std::vector<int>> &occupancy,
                     const Wire &wire,
                     const std::vector<uint8_t> &mask, int dim_x) {
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
    int dx = (x_end > x) ? 1 : (x_end < x) ? -1 : 0;
    int dy = (y_end > y) ? 1 : (y_end < y) ? -1 : 0;
    while (x != x_end || y != y_end) {
      int v = occupancy[y][x] - mask[y * dim_x + x];
      cost += v * v;
      x += dx; y += dy;
    }
    if (seg == num_pts - 2) {
      int v = occupancy[y][x] - mask[y * dim_x + x];
      cost += v * v;
    }
  }
  return cost;
}

Wire build_route(const Wire &wire, int route_index) {
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

// --- RouteUpdate: convert Wire to RouteUpdate for wire_id ---
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

// --- Peer-to-peer exchange: send my updates to all others, receive from all, apply ---
// Pack: [int count][RouteUpdate data]. One contiguous send/recv per peer.
void exchange_updates(std::vector<std::vector<int>> &occupancy,
                      std::vector<Wire> &wires,
                      const std::vector<RouteUpdate> &my_updates,
                      int dim_x, int dim_y, int rank, int nproc,
                      std::vector<std::vector<char>> &recv_bufs) {

  const int tag = 0;
  const int max_count = dim_x * dim_y * 2;
  const int max_bytes = static_cast<int>(sizeof(int)) + max_count * static_cast<int>(sizeof(RouteUpdate));

  std::vector<char> send_buf(max_bytes);
  int count = static_cast<int>(my_updates.size());
  std::memcpy(send_buf.data(), &count, sizeof(int));
  if (count > 0) {
    std::memcpy(send_buf.data() + sizeof(int), my_updates.data(), count * sizeof(RouteUpdate));
  }
  int send_size = static_cast<int>(sizeof(int)) + count * static_cast<int>(sizeof(RouteUpdate));

  std::vector<MPI_Request> reqs;
  reqs.reserve(2 * (nproc - 1));

  for (int p = 0; p < nproc; p++) {
    if (p == rank) continue;
    MPI_Request r;
    MPI_Isend(send_buf.data(), send_size, MPI_BYTE, p, tag, MPI_COMM_WORLD, &r);
    reqs.push_back(r);
  }

  for (int p = 0; p < nproc; p++) {
    if (p == rank) continue;
    MPI_Request r;
    MPI_Irecv(recv_bufs[p].data(), max_bytes, MPI_BYTE, p, tag, MPI_COMM_WORLD, &r);
    reqs.push_back(r);
  }

  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for (int p = 0; p < nproc; p++) {
    if (p == rank) continue;
    int n;
    std::memcpy(&n, recv_bufs[p].data(), sizeof(int));
    const RouteUpdate *updates = reinterpret_cast<const RouteUpdate *>(recv_bufs[p].data() + sizeof(int));
    for (int k = 0; k < n; k++) {
      const RouteUpdate &u = updates[k];
      int wi = u.wire_id;
      if (wi < 0 || wi >= static_cast<int>(wires.size())) continue;
      Wire old_wire = wires[wi];
      update_occupancy(occupancy, old_wire, -1);
      wires[wi] = update_to_wire(u);
      update_occupancy(occupancy, wires[wi], +1);
    }
  }
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

  int opt;
  while ((opt = getopt(argc, argv, "f:n:p:i:m:b:")) != -1) {
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
      default:
        if (rank == 0) {
          std::cerr << "Usage: " << argv[0]
                    << " -f input_filename -n num_procs [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  if (input_filename.empty() || n_flag <= 0 || SA_iters <= 0 || batch_size <= 0 ||
      (parallel_mode != 'A' && parallel_mode != 'W')) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0]
                << " -f input_filename -n num_procs [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (n_flag != nproc) {
    if (rank == 0) {
      std::cerr << "Error: -n " << n_flag << " must match number of MPI processes " << nproc << "\n";
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (parallel_mode != 'A') {
    if (rank == 0) {
      std::cerr << "This implementation supports only across-wires mode (-m A).\n";
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
  }

  int dim_x, dim_y, num_wires;
  std::vector<Wire> wires;
  std::vector<std::vector<int>> occupancy;

  {
    std::ifstream fin(input_filename);
    if (!fin) {
      std::cerr << "Rank " << rank << ": unable to open " << input_filename << '\n';
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
    fin >> dim_x >> dim_y >> num_wires;
    wires.resize(num_wires);
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
  }

  occupancy.assign(dim_y, std::vector<int>(dim_x, 0));
  for (const auto &wire : wires) {
    update_occupancy(occupancy, wire, +1);
  }

  if (rank == 0) {
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';
  }

  const auto compute_start = std::chrono::steady_clock::now();

  std::vector<uint8_t> cell_mask(dim_y * dim_x, 0);
  std::mt19937 rng(42 + rank);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  const int max_updates_per_peer = dim_x * dim_y * 2;
  const size_t max_exchange_bytes = sizeof(int) + max_updates_per_peer * sizeof(RouteUpdate);
  std::vector<std::vector<char>> recv_bufs(nproc);
  for (int p = 0; p < nproc; p++) {
    recv_bufs[p].resize(max_exchange_bytes);
  }

  for (int iter = 0; iter < SA_iters; iter++) {
    for (int batch_start = 0; batch_start < num_wires; batch_start += batch_size) {
      int batch_end = std::min(batch_start + batch_size, num_wires);
      std::vector<RouteUpdate> my_updates;

      for (int wi = batch_start; wi < batch_end; wi++) {
        if (wi % nproc != rank) continue;

        Wire &wire = wires[wi];
        int abs_dx = std::abs(wire.end_x - wire.start_x);
        int abs_dy = std::abs(wire.end_y - wire.start_y);
        if (abs_dx == 0 || abs_dy == 0) continue;

        int total_routes = abs_dx + abs_dy + 2 * (abs_dx - 1) * (abs_dy - 1);

        if (prob_dist(rng) < SA_prob) {
          std::uniform_int_distribution<int> route_dist(0, total_routes - 1);
          Wire new_wire = build_route(wire, route_dist(rng));
          update_occupancy(occupancy, wire, -1);
          wire = new_wire;
          update_occupancy(occupancy, wire, +1);
          my_updates.push_back(wire_to_update(wi, wire));
          continue;
        }

        mask_wire_cells(wire, cell_mask, dim_x, 1);
        int best_cost = path_cost_masked(occupancy, wire, cell_mask, dim_x);
        int best_idx = -1;

        for (int r = 0; r < total_routes; r++) {
          Wire candidate = build_route(wire, r);
          int cost = path_cost_masked(occupancy, candidate, cell_mask, dim_x);
          if (cost < best_cost) {
            best_cost = cost;
            best_idx = r;
          }
        }
        mask_wire_cells(wire, cell_mask, dim_x, 0);

        if (best_idx >= 0) {
          Wire new_wire = build_route(wire, best_idx);
          update_occupancy(occupancy, wire, -1);
          wire = new_wire;
          update_occupancy(occupancy, wire, +1);
          my_updates.push_back(wire_to_update(wi, wire));
        }
      }

      exchange_updates(occupancy, wires, my_updates, dim_x, dim_y, rank, nproc,
                       recv_bufs);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
  }

  if (rank == 0) {
    wr_checker checker(wires, occupancy);
    checker.validate();
    print_stats(occupancy);
    write_output(wires, num_wires, occupancy, dim_x, dim_y,
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
