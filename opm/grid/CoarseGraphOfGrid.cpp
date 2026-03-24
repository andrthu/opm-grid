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

#include <config.h>
#include "CoarseGraphOfGrid.hpp"

#include <numeric>


namespace Opm {

template<typename Grid>
void CoarseGraphOfGrid<Grid>::dfs(Row row, int v, int master, double w, std::vector<bool>& visited,
                            std::vector<int>& cnode, std::vector<std::tuple<int,int,double> >& edges)
{
    visited[v] = true;
	f2c[v] = master;
	cnode.push_back(v);
	
	auto col = row.begin();
	for (; col != row.end(); ++col) {
	    int nab = col.index();
        
	    if ((*transGraph)[v][nab] > w) {
            if (!visited[nab]) {
                dfs((*transGraph)[nab],nab,master,w,visited,cnode,edges);
            } else {
                if (f2c[v]!=f2c[nab]) {
                    std::cout << "Problem " << nab << " " << v <<
                        " " << f2c[v] << " " << f2c[nab] <<std::endl; 
                }
            }
	    }
	}
	col = row.begin();
	for (; col != row.end(); ++col) {
	    int nab = col.index();
	    if (f2c[v]!=f2c[nab]) {
            edges.push_back({v,nab,(*transGraph)[v][nab]});
	    }
	}
}

template<typename Grid>
void CoarseGraphOfGrid<Grid>::createCoarseGraph(const double* transmissibilities,
                                          const Dune::EdgeWeightMethod edgeWeightMethod,
                                          double coarseThreshold)
{
    int N = grid.size(0);
    const auto& rank = grid.comm().rank();
    if (rank == 0) {std::cout << "Start create coarse graph" << std::endl;}

    
    std::vector<bool> visited(N, false);
    //std::vector<int> f2c;
    f2c.resize(N, 0);
    std::vector<int> c2f;
    
    std::vector<std::vector<std::tuple<int,int,double> >> gEdges;

    int newV = 0;

    int biggest = 0;
    if (rank == 0) {
        for (int v = 0; v < N; ++v) {

            if (!visited[v]) {

                c2f.push_back(v);
                std::vector<int> cnode;
                std::vector<std::tuple<int,int,double> > edges;
                dfs((*transGraph)[v],v,newV,coarseThreshold,visited,cnode,edges);
                newV++;
                gEdges.push_back(edges);
                coarseNodes.push_back(cnode);
                if ((int)cnode.size() > biggest)
                    biggest = cnode.size();
            }
        }
        std::cout << "Coarse graph size " << coarseNodes.size() <<" "<< biggest << std::endl;
        //std::vector<std::map<int, double> > cedges;
        for (std::vector<std::tuple<int,int,double> > es : gEdges ) {
            std::map<int, double> ce;
            for (std::tuple<int,int,double> fe : es) {
                int coarseNab = f2c[std::get<1>(fe)];
                double transVal = std::get<2>(fe);
                double weight = edgeWeightMethod == 0 ? 1.0 : transVal;
                if (transVal > 0) {
                    if ( ce.count(coarseNab) == 1 ) {
                        ce[coarseNab] += weight;
                    } else {
                        ce.insert({coarseNab,weight});
                    }
                }
            }
            cedges.push_back(ce);
        }
        std::cout << "Coarse graph size edges " << gEdges.size() << std::endl;
    }
}

template<typename Grid>
void CoarseGraphOfGrid<Grid>::dfsq(Row row, std::priority_queue<WgtIdx2> &q, int v, int master,
                             double w, int maxNode, std::vector<bool>& visited,
                             std::vector<int>& cnode, std::vector<std::tuple<int,int,double> >& edges)
{
    visited[v] = true;
	f2c[v] = master;
	cnode.push_back(v);
	
	auto col = row.begin();
	for (; col != row.end(); ++col) {
	    int nab = col.index();
        double wgt = (*transGraph)[v][nab];
	    if ( wgt > w) {
            if (!visited[nab]) {
                q.push({wgt, nab});
                //dfsq((*transGraph)[nab],q,nab,master,w,maxNode,visited,cnode,edges);
            } 
	    }
	}

    if ( (int)cnode.size() < maxNode ) {
        if (!q.empty()) {

            auto strongCon = q.top();
            int nab = strongCon.idx;
            q.pop();
            while (visited[nab] && !q.empty()) {
                strongCon = q.top();
                nab = strongCon.idx;
                q.pop();
            }
            if (!visited[nab])
                dfsq((*transGraph)[nab],q,nab,master,w,maxNode,visited,cnode,edges);
        }
    } else {

        q = std::priority_queue<WgtIdx2>();
    }

	col = row.begin();
	for (; col != row.end(); ++col) {
	    int nab = col.index();
	    if (f2c[v]!=f2c[nab]) {
            edges.push_back({v,nab,(*transGraph)[v][nab]});
	    }
	}
}

template<typename Grid>
void CoarseGraphOfGrid<Grid>::createCoarseGraph(const double* transmissibilities,
                                          const Dune::EdgeWeightMethod edgeWeightMethod,
                                          double coarseThreshold,
                                          int coarsePartitionMaxNodeSize)
{
    int N = grid.size(0);
    const auto& rank = grid.comm().rank();
    if (rank == 0) {std::cout << "Start create coarse graph" << std::endl;}

    
    std::vector<bool> visited(N, false);
    //std::vector<int> f2c;
    f2c.resize(N, 0);
    std::vector<int> c2f;
    
    std::vector<std::vector<std::tuple<int,int,double> >> gEdges;

    int newV = 0;

    int biggest = 0;
    if (rank == 0) {
        for (int v = 0; v < N; ++v) {

            if (!visited[v]) {

                std::priority_queue<WgtIdx2> q;
                c2f.push_back(v);
                std::vector<int> cnode;
                std::vector<std::tuple<int,int,double> > edges;
                dfsq((*transGraph)[v],q,v,newV,coarseThreshold,
                     coarsePartitionMaxNodeSize,visited,cnode,edges);
                newV++;
                gEdges.push_back(edges);
                coarseNodes.push_back(cnode);
                if ((int)cnode.size() > biggest)
                    biggest = cnode.size();
            }
        }
        std::cout << "Coarse maxNodeSize graph size " << coarseNodes.size() <<" "<< biggest << std::endl;
        //std::vector<std::map<int, double> > cedges;
        for (std::vector<std::tuple<int,int,double> > es : gEdges ) {
            std::map<int, double> ce;
            for (std::tuple<int,int,double> fe : es) {
                int coarseNab = f2c[std::get<1>(fe)];
                double transVal = std::get<2>(fe);
                double weight = edgeWeightMethod == 0 ? 1.0 : transVal;
                if (transVal > 0) {
                    if ( ce.count(coarseNab) == 1 ) {
                        ce[coarseNab] += weight;
                    } else {
                        ce.insert({coarseNab,weight});
                    }
                }
            }
            cedges.push_back(ce);
        }
        std::cout << "Coarse graph size edges " << gEdges.size() << std::endl;
    }
}
template<typename Grid>
void CoarseGraphOfGrid<Grid>::mergeWellCellsForCoarseGraph(std::vector<int>& hasWell,
                                                     std::vector<std::vector<int>>& wellPerf,
                                                     const Dune::cpgrid::WellConnections& wellConn)
{
    int wellId = 0;

    for (const auto& well : wellConn) {

        bool cellInMultWells = false;
        int otherWell = -1;
        std::vector<int> perfs;
        for (int idx : well) {

            if (!cellInMultWells) {
                if (hasWell[idx]!=-1) {
                    cellInMultWells = true;
                    otherWell = hasWell[idx];
                } else {
                    hasWell[idx] = wellId;
                    perfs.push_back(idx);
                }
            }
        }
        if (cellInMultWells) {
            for (int idx : well) {
                hasWell[idx] = otherWell;
                wellPerf[otherWell].push_back(idx);
            }
        }
        else {
            wellPerf.push_back(perfs);
            wellId++;
        }
    }
}

template<typename Grid>
void CoarseGraphOfGrid<Grid>::dfsqw(Row row, std::priority_queue<WgtIdx2> &q, int v, int master,
                              double w, int maxNode, std::vector<bool>& visited,
                              std::vector<int>& cnode, std::vector<std::tuple<int,int,double> >& edges,
                              std::vector<int>& hasWell, std::vector<std::vector<int>>& wellPerf)
{
    if (hasWell[v] == -1) {
        visited[v] = true;
        f2c[v] = master;
        cnode.push_back(v);
        
        auto col = row.begin();
        for (; col != row.end(); ++col) {
            int nab = col.index();
            double wgt = (*transGraph)[v][nab];
            if ( wgt > w) {
                if (!visited[nab]) {
                    q.push({wgt, nab});
                    //dfsq((*transGraph)[nab],q,nab,master,w,maxNode,visited,cnode,edges);
                }
            }
        }
    } else {
        int wellId = hasWell[v];
        std::vector<int> perfs = wellPerf[wellId];

        for (const auto& idx : perfs) {
            visited[idx] = true;
            f2c[idx] = master;
            cnode.push_back(idx);
        }
        for (const auto& idx : perfs) {
            auto wrow = (*transGraph)[idx];
            auto col = wrow.begin();
            for (; col != wrow.end(); ++col) {
                int nab = col.index();
                double wgt = (*transGraph)[idx][nab];
                if ( wgt > w) {
                    if (!visited[nab]) {
                        q.push({wgt, nab});
                        //dfsq((*transGraph)[nab],q,nab,master,w,maxNode,visited,cnode,edges);
                    }
                }
            }
        }
    }

    if ( (int)cnode.size() < maxNode ) {
        if (!q.empty()) {

            auto strongCon = q.top();
            int nab = strongCon.idx;
            q.pop();
            while (visited[nab] && !q.empty()) {
                strongCon = q.top();
                nab = strongCon.idx;
                q.pop();
            }
            if (!visited[nab])
                dfsqw((*transGraph)[nab],q,nab,master,w,maxNode,visited,cnode,edges,hasWell,wellPerf);
        }
    } else {

        q = std::priority_queue<WgtIdx2>();
    }

	auto col = row.begin();
	for (; col != row.end(); ++col) {
	    int nab = col.index();
	    if (f2c[v]!=f2c[nab]) {
            edges.push_back({v,nab,(*transGraph)[v][nab]});
	    }
	}
}

template<typename Grid>
void CoarseGraphOfGrid<Grid>::createCoarseGraph(const double* transmissibilities,
                                          const Dune::EdgeWeightMethod edgeWeightMethod,
                                          double coarseThreshold,
                                          int coarsePartitionMaxNodeSize,
                                          const Dune::cpgrid::WellConnections& wellConn)
{
    int N = grid.size(0);
    const auto& rank = grid.comm().rank();
    if (rank == 0) {std::cout << "Start create coarse graph" << std::endl;}


    std::vector<bool> visited(N, false);
    std::vector<int> hasWell(N,-1);
    std::vector<std::vector<int>> wellPerf;
    mergeWellCellsForCoarseGraph(hasWell, wellPerf, wellConn);
    //std::vector<int> f2c;
    f2c.resize(N, 0);
    std::vector<int> c2f;

    std::vector<std::vector<std::tuple<int,int,double> >> gEdges;

    int newV = 0;

    int biggest = 0;
    if (rank == 0) {
        for (int v = 0; v < N; ++v) {

            if (!visited[v]) {

                std::priority_queue<WgtIdx2> q;
                c2f.push_back(v);
                std::vector<int> cnode;
                std::vector<std::tuple<int,int,double> > edges;
                dfsqw((*transGraph)[v],q,v,newV,coarseThreshold,
                      coarsePartitionMaxNodeSize,visited,cnode,edges,
                      hasWell, wellPerf);
                newV++;
                gEdges.push_back(edges);
                coarseNodes.push_back(cnode);
                if ((int)cnode.size() > biggest)
                    biggest = cnode.size();
            }
        }
        std::cout << "Coarse maxNodeSize graph size " << coarseNodes.size() <<" "<< biggest << std::endl;
        //std::vector<std::map<int, double> > cedges;
        for (std::vector<std::tuple<int,int,double> > es : gEdges ) {
            std::map<int, double> ce;
            for (std::tuple<int,int,double> fe : es) {
                int coarseNab = f2c[std::get<1>(fe)];
                double transVal = std::get<2>(fe);
                double weight = edgeWeightMethod == 0 ? 1.0 : transVal;
                if (transVal > 0) {
                    if ( ce.count(coarseNab) == 1 ) {
                        ce[coarseNab] += weight;
                    } else {
                        ce.insert({coarseNab,weight});
                    }
                }
            }
            cedges.push_back(ce);
        }
        std::cout << "Coarse graph size edges " << gEdges.size() << std::endl;
    }
}
    
template class CoarseGraphOfGrid<Dune::CpGrid>;

} // namespace Opm
