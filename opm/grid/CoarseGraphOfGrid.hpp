// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2024 Equinor ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/

#ifndef OPM_GRAPH_OF_GRID_HEADER
#define OPM_GRAPH_OF_GRID_HEADER

#include <opm/grid/CpGrid.hpp>
#include <opm/grid/common/WellConnections.hpp>
#include <queue>

namespace Opm {

struct WgtIdx {

    double wgt;
    int idx;

    bool operator<(const WgtIdx& other) const {
	return wgt < other.wgt;
    }
};

/// \brief A class storing a graph representation of the grid
///
/// Stores the list of all cell global IDs and for each cell
/// a list of global IDs of its neighbors.
/// In addition, weights of graph vertices and edges are stored.
///
/// Features edge contractions, which adds weights of merged vertices
/// and of edges to every shared neighbor. Intended use is for loadbalancing
/// to ensure that no well is split between processes.
template<typename Grid>
class CoarseGraphOfGrid{
    using WeightType = float;
    using EdgeList = std::map<int,WeightType>;

    struct VertexProperties
    {
        int nproc = 0; // number of processor
        WeightType weight = 1; // vertex weight
        EdgeList edges;
    };

    using TransGraph = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
    using Row = typename TransGraph::row_type;

public:

    explicit CoarseGraphOfGrid (const Grid& grid_,
                          const double* transmissibilities,
                          const Dune::EdgeWeightMethod edgeWeightMethod,
                          TransGraph* tg,
                          double coarseThreshold,
                          int coarsePartitionMaxNodeSize,
                          bool allowDistributedWells,
                          const Dune::cpgrid::WellConnections& wellConn)
        : grid(grid_), transGraph(tg)
    {
        if (allowDistributedWells) {
            if (coarsePartitionMaxNodeSize == -1)
                createCoarseGraph(transmissibilities, edgeWeightMethod, coarseThreshold);
            else
                createCoarseGraph(transmissibilities, edgeWeightMethod, coarseThreshold, coarsePartitionMaxNodeSize);
        }
        else {
            if (coarsePartitionMaxNodeSize == -1) {
                std::cout << "Merging wells only supported with coarsePartitionMaxNodeSize!=-1" << std::endl; 
                createCoarseGraph(transmissibilities, edgeWeightMethod, coarseThreshold);
            }
            else
                createCoarseGraph(transmissibilities, edgeWeightMethod, coarseThreshold, coarsePartitionMaxNodeSize, wellConn);
        }
    }

    const Grid& getGrid() const
    {
        return grid;
    }

    int cSize() const
    {
        return coarseNodes.size();
    }

    std::vector<std::vector<int>> getCoarseNodes() const
    {
        return coarseNodes;
    }

    std::vector<std::map<int, double> > getCoarseEdges() const
    {
        return cedges;
    }

    std::vector<int> getF2c() const
    {
        return f2c;
    }

    /// \brief Return the list of wells
    const auto& getWells () const
    {
        return wells;
    }
private:
    void dfs(Row row, int v, int master, double w, std::vector<bool>& visited,
             std::vector<int>& cnode, std::vector<std::tuple<int,int,double> >& edges);

    void createCoarseGraph(const double* transmissibilities,
                           const Dune::EdgeWeightMethod edgeWeightMethod,
                           double coarseThreshold);

    void dfsq(Row row, std::priority_queue<WgtIdx> &q, int v, int master,
              double w, int maxNode, std::vector<bool>& visited,
              std::vector<int>& cnode, std::vector<std::tuple<int,int,double> >& edges);

    void createCoarseGraph(const double* transmissibilities,
                           const Dune::EdgeWeightMethod edgeWeightMethod,
                           double coarseThreshold,
                           int coarsePartitionMaxNodeSize);

    void mergeWellCellsForCoarseGraph(std::vector<int>& hasWell,
                                      std::vector<std::vector<int>>& wellPerf,
                                      const Dune::cpgrid::WellConnections& wells);

    void dfsqw(Row row, std::priority_queue<WgtIdx> &q, int v, int master,
               double w, int maxNode, std::vector<bool>& visited,
               std::vector<int>& cnode, std::vector<std::tuple<int,int,double> >& edges,
               std::vector<int>& hasWell, std::vector<std::vector<int>>& wellPerf);

    void createCoarseGraph(const double* transmissibilities,
                           const Dune::EdgeWeightMethod edgeWeightMethod,
                           double coarseThreshold,
                           int coarsePartitionMaxNodeSize,
                           const Dune::cpgrid::WellConnections& wells);

    const Grid& grid;
    std::list<std::set<int>> wells;

    Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>* transGraph;
    std::vector<int> f2c;
    std::vector<std::map<int, double> > cedges;
    std::vector<std::vector<int>> coarseNodes;
    
};

} // namespace Opm

#endif // OPM_GRAPH_OF_GRID_HEADER
