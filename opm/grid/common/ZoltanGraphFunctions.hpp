/*
  Copyright 2015 Dr. Blatt - HPC-Simulation-Software & Services.
  Copyright 2015 NTNU
  Copyright 2015 Statoil AS

  This file is part of The Open Porous Media project  (OPM).

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
*/
#ifndef DUNE_CPGRID_ZOLTAN_GRAPH_FUNCTIONS_HEADER
#define DUNE_CPGRID_ZOLTAN_GRAPH_FUNCTIONS_HEADER

#include <opm/grid/utility/OpmParserIncludes.hpp>

#include <opm/grid/CpGrid.hpp>
#include <opm/grid/common/WellConnections.hpp>

#if defined(HAVE_ZOLTAN) && defined(HAVE_MPI)

#include <algorithm>
#include <mpi.h>

// Zoltan redefines HAVE_MPI. Therefore we need to back it up, undef, and
// redifine it after the header is included
#undef HAVE_MPI
#include <zoltan.h>
#undef HAVE_MPI
#define HAVE_MPI 1

namespace Dune
{
namespace cpgrid
{
/// \brief Get the number of cells of the grid.
///
/// The cells are the vertices of the graph.
/// \return The number of vertices of the graph representing the grid.
inline int getCpGridNumCells(void* cpGridPointer, int* err)
{
    const Dune::CpGrid&  grid = *static_cast<const Dune::CpGrid*>(cpGridPointer);
    *err = ZOLTAN_OK;
    return grid.numCells();
}

/// \brief Get the list of vertices of the graph of the grid.
void getCpGridVertexList(void* cpGridPointer, int numGlobalIds,
                         int numLocalIds, ZOLTAN_ID_PTR gids,
                         ZOLTAN_ID_PTR lids, int wgtDim,
                         float *objWgts, int *err);

/// \brief Get the number of edges the graph of the grid.
void getCpGridNumEdgesList(void *cpGridPointer, int sizeGID, int sizeLID,
                           int numCells,
                           ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                           int *numEdges, int *err);

/// \brief Get the list of edges of the graph of the grid.
void getCpGridEdgeList(void *cpGridPointer, int sizeGID, int sizeLID,
                       int numCells, ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                       int *num_edges,
                       ZOLTAN_ID_PTR nborGID, int *nborProc,
                       int wgt_dim, float *ewgts, int *err);

/// \brief Get a list of vertices with zero enties
void getNullVertexList(void* cpGridPointer, int numGlobalIds,
                         int numLocalIds, ZOLTAN_ID_PTR gids,
                         ZOLTAN_ID_PTR lids, int wgtDim,
                         float *objWgts, int *err);

/// \brief Get zero as the number of edges the graph of the grid.
void getNullNumEdgesList(void *cpGridPointer, int sizeGID, int sizeLID,
                           int numCells,
                           ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                           int *numEdges, int *err);

/// \brief Get a list of edges of size zero.
void getNullEdgeList(void *cpGridPointer, int sizeGID, int sizeLID,
                       int numCells, ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                       int *num_edges,
                       ZOLTAN_ID_PTR nborGID, int *nborProc,
                       int wgt_dim, float *ewgts, int *err);

/// \brief Get always zero as the number of cells of the grid.
///
/// The cells are the vertices of the graph.
inline int getNullNumCells(void* cpGridPointer, int* err)
{
    (void) cpGridPointer;
    *err = ZOLTAN_OK;
    return 0;
}

/// \brief Get the number of edges the graph of the grid and the wells.
void getCpGridWellsNumEdgesList(void *cpGridWellsPointer, int sizeGID, int sizeLID,
                           int numCells,
                           ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                           int *numEdges, int *err);

/// \brief Get the list of edges of the graph of the grid and the wells
void getCpGridWellsEdgeList(void *cpGridWellsPointer, int sizeGID, int sizeLID,
                       int numCells, ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                       int *num_edges,
                       ZOLTAN_ID_PTR nborGID, int *nborProc,
                       int wgt_dim, float *ewgts, int *err);

/// \brief A graph repesenting a grid together with the well completions.
///
/// The edges of the graph are formed by the superset of the edges representing
/// the faces of the grid and the ones that represent the connections within the
/// wells. If a well has completions
/// on cell i and cell j, then there is an edge from i to j and j to i in the graph.
/// Even for shut wells the connections will exist.
class CombinedGridWellGraph
{
public:
    typedef std::vector<std::set<int> > GraphType;

    /// \brief Create a graph representing a grid together with the wells.
    /// \param grid The grid.
    /// \param eclipseState The eclipse state to extract the well information from.
    /// \param pretendEmptyGrid True if we should pretend the grid and wells are empty.
    CombinedGridWellGraph(const Dune::CpGrid& grid,
                          const std::vector<const OpmWellType*> * wells,
                          const double* transmissibilities,
                          bool pretendEmptyGrid, int edgeWeightsMethod, 
			  bool useObjWgt, std::vector<double> catW);

    /// \brief Access the grid.
    const Dune::CpGrid& getGrid() const
    {
        return grid_;
    }

    const GraphType& getWellsGraph() const
    {
        return wellsGraph_;
    }
    
    const std::vector<int>& getVertexWeights() const
    {
	return vertexWeightsWithWells_;
    }

    const std::vector<int>& getVertexWeightsWithWells() const
    {
	return vertexWeightsWithWells_;
    }

    double transmissibility(int face_index) const
    {
	return transmissibilities_ ? 1.0e18*transmissibilities_[face_index] : 1.0;
    }

    double logTransmissibilityWeights(int face_index) const
    {
	double trans = transmissibilities_[face_index]; 
	return trans == 0.0 ? 0.0 : 1.0 + std::log(trans) - log_min_;
    }

    double logTransmissibilityWeights2(int face_index) const
    {
	double trans = transmissibilities_[face_index];
	return trans == 0.0 ? 0.1 : 1.0 + std::log(trans) - log_min_;
    }

    double catagoryTransWeights(int face_index) const
    {
	double trans = transmissibilities_[face_index];
	if (trans == 0.0) {return 0.0;}
	if (trans < trans_bound_[0]) {return catW_[0];}
	if (trans < trans_bound_[1]) {return catW_[1];}
	return catW_[2];
    }

    double normalizedTransWeights(int face, int cell0, int cell1) const
    {
	double trans = transmissibilities_[face];
	double d0 = diag_normalized_[cell0];
	double d1 = diag_normalized_[cell1];
	return trans/std::sqrt(d0*d1);
    }

    double uniformWeights(int face) const
    {
	if (transmissibilities_[face] == 0.0) {return 0;}
	return 1.0;
    }

    double edgeWeight(int face_index, int cell1, int cell2) const
    {
	if (edgeWeightsMethod_ == 0)
	    return uniformWeights(face_index);
	else if (edgeWeightsMethod_ == 1)
	    return transmissibility(face_index);
	else if (edgeWeightsMethod_ == 2)
	    return logTransmissibilityWeights(face_index);
	else if (edgeWeightsMethod_ == 3)
	    return logTransmissibilityWeights2(face_index);
	else if (edgeWeightsMethod_ == 4)
	    return catagoryTransWeights(face_index);
	else if (edgeWeightsMethod_ == 5)
	    return normalizedTransWeights(face_index, cell1, cell2);
	else
	    return 1.0;
    }

    const WellConnections& getWellConnections() const
    {
        return well_indices_;
    }
private:

    void addCompletionSetToGraph()
    {
        for(const auto& well_indices : well_indices_)
        {
            for( auto well_idx = well_indices.begin(); well_idx != well_indices.end();
                 ++well_idx)
            {
                auto well_idx2 = well_idx;
                for( ++well_idx2; well_idx2 != well_indices.end();
                     ++well_idx2)
                {
                    wellsGraph_[*well_idx].insert(*well_idx2);
                    wellsGraph_[*well_idx2].insert(*well_idx);
                }
            }
        }
    }
    void findDiagNormalized()
    {
	int N = grid_.numCells();
	diag_normalized_.resize(N, 0.0);
	for (int cell = 0; cell < N; ++cell)
	{
	    double diag_val = 0;
	    for ( int local_face = 0; local_face < grid_.numCellFaces(cell); ++local_face )
	    {
		const int face  = grid_.cellFace(cell, local_face);
		const int face0 = grid_.faceCell(face, 0);
		const int face1 = grid_.faceCell(face, 1);
		if ( face0 != -1 && face1 != -1 )
		{
		    diag_val += transmissibilities_[face];
		}
	    }
	    diag_normalized_[cell] = diag_val;
	}
    }

    void findMaxMinTrans()
    {
	double min_val = std::numeric_limits<float>::max();
		
	for (int face = 0; face < getGrid().numFaces(); ++face)
	{
	    double trans = transmissibilities_[face];
	    if (trans > 0)
	    {
		if (trans < min_val)
		    min_val = trans;		
	    }
	}	
	log_min_ = std::log(min_val);
    }

    void calculateVertexWeights()
    {
	auto& globalIdSet = grid_.globalIdSet();
	vertexWeights_.resize(grid_.numCells(), 0);
	vertexWeightsWithWells_.resize(grid_.numCells(), 0);

	int idx = 0;
	for (auto cell = grid_.leafbegin<0>(); cell != grid_.leafend<0>();++cell)
	{
	    int cid = globalIdSet.id(*cell);
	    vertexWeights_[idx] = grid_.numCellFaces(cid) + 1;
	    vertexWeightsWithWells_[idx] = vertexWeights_[idx] + wellsGraph_[cid].size();

	    idx++;
	}
    }

    void sortTrans()
    {
	int numFace = grid_.numFaces();
	std::vector<double> trans_sort(transmissibilities_, transmissibilities_ + numFace);

	std::sort(trans_sort.begin(), trans_sort.end());

	trans_bound_.push_back(trans_sort[numFace/4]);
	trans_bound_.push_back(trans_sort[3*(numFace/4)]);
	if (grid_.comm().rank() == 0) {
	    std::cout << "Bounds for trans weights: "; 
	    for (unsigned i = 0; i < trans_bound_.size(); ++i)
		std::cout << trans_bound_[i] << " ";
	    std::cout << std::endl;
	}
    }

    const Dune::CpGrid& grid_;
    GraphType wellsGraph_;
    const double* transmissibilities_;
    WellConnections well_indices_;
    std::vector<int> vertexWeights_;
    std::vector<int> vertexWeightsWithWells_;
    std::vector<double> trans_bound_;
    std::vector<double> diag_normalized_;

    int edgeWeightsMethod_;
    double log_min_;

    std::vector<double> catW_;
};


/// \brief Sets up the call-back functions for ZOLTAN's graph partitioning.
/// \param zz The struct with the information for ZOLTAN.
/// \param grid The grid to partition.
/// \param pretendNull If true, we will pretend that the grid has zero cells.
void setCpGridZoltanGraphFunctions(Zoltan_Struct *zz, const Dune::CpGrid& grid,
                                   bool pretendNull=false);

void setCpGridZoltanGraphFunctions(Zoltan_Struct *zz,
                                   const CombinedGridWellGraph& graph,
                                   bool pretendNull);

void getCpGridHyperGraphSize(void *graphPointer, int* num_lists,
			     int *num_pins, int *format, int *err);

void getNullHyperGraphSize(void *graphPointer, int* num_lists,
			   int *num_pins, int *format, int *err);

void getCpGridHyperGraphList(void *graphPointer, int num_gid_entries, 
			     int num_vtx_edge, int num_pins, int format,
			     ZOLTAN_ID_PTR vtxedge_GID, int *vtxedge_ptr, 
			     ZOLTAN_ID_PTR pin_GID, int *err);

void getNullHyperGraphList(void *graphPointer, int num_gid_entries, 
			   int num_vtx_edge, int num_pins, int format,
			   ZOLTAN_ID_PTR vtxedge_GID, int *vtxedge_ptr,
			   ZOLTAN_ID_PTR pin_GID, int *err);

void getCpGridHyperGraphWgtSize(void *graphPointer, int *num_edges, int *err);

void getNullHyperGraphWgtSize(void *graphPointer, int *num_edges, int *err);

void getCpGridHyperGraphWgtVal(void *graphPointer, int num_gid_entries, 
			       int num_lid_entries, int num_edges, 
			       int edge_weight_dim, ZOLTAN_ID_PTR edge_GID, 
			       ZOLTAN_ID_PTR edge_LID, float  *edge_weight, int *err);

void getNullHyperGraphWgtVal(void *graphPointer, int num_gid_entries, 
			     int num_lid_entries, int num_edges, 
			     int edge_weight_dim, ZOLTAN_ID_PTR edge_GID, 
			     ZOLTAN_ID_PTR edge_LID, float  *edge_weight, int *err);

void setCpGridZoltanHyperGraphFunctions(Zoltan_Struct *zz,
					const CombinedGridWellGraph& graph,
					bool pretendNull, bool useWgt);

} // end namespace cpgrid
} // end namespace Dune

#endif // HAVE_ZOLTAN
#endif // header guard
