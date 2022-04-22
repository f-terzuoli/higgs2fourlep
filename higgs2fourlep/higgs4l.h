#ifndef HIGGSFOURLEP_H
#define HIGGSFOURLEP_H

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1D.h"
#include "TLatex.h"
#include "TMath.h"
#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"
#include "Math/Plane3D.h"
#include "TStyle.h"
#include <iostream>
#include <vector>

using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;
using rvec_f = const RVec<float> &;
using rvec_i = const RVec<int> &;
const auto z_mass = 91.2;

//Selecting leptons with high transverse momentum, within geometrical acepptance, with low long. IPs and  low transverse IP_CHI2
bool SelectElectronsAndMuons(rvec_i type, rvec_f pt, rvec_f eta, rvec_f phi, rvec_f e, rvec_f tracksigd0pv, rvec_f z0)
{
    for (size_t i = 0; i < type.size(); i++) {
        ROOT::Math::PtEtaPhiEVector p(pt[i] / 1000.0, eta[i], phi[i], e[i] / 1000.0);
        if (type[i] == 11) {
            if (pt[i] < 7000 || abs(eta[i]) > 2.47 || abs(tracksigd0pv[i]) > 5 || abs(z0[i] * sin(p.Theta())) > 0.5) return false;
        } else {
            if (abs(tracksigd0pv[i]) > 3 || abs(z0[i] * sin(p.Theta())) > 0.5) return false;
        }
    }
    return true;
}


// Identify the leading and subleading lepton pair: idx[0][i]-->leading and idx[1][i]-->subleading
RVec<RVec<size_t>> reco_zz(rvec_f pt, rvec_f eta, rvec_f phi, rvec_f e, rvec_i charge, rvec_i type, int sumtypes)
{
   RVec<RVec<size_t>> idx(2);
   idx[0].reserve(2); idx[1].reserve(2);

   // Find first lepton pair with invariant mass closest to Z mass
   auto idx_cmb = Combinations(pt, 2);
   double best_mass = -1e6;
   size_t best_i1 = 0; size_t best_i2 = 0;
    if (sumtypes == 48) {
        for (size_t i = 0; i < idx_cmb[0].size(); i++) {
           const auto i1 = idx_cmb[0][i];
           const auto i2 = idx_cmb[1][i];
           if (charge[i1] * charge[i2] < 0 && type[i1] == type[i2]) {
              ROOT::Math::PtEtaPhiEVector p1(pt[i1], eta[i1], phi[i1], e[i1]);
              ROOT::Math::PtEtaPhiEVector p2(pt[i2], eta[i2], phi[i2], e[i2]);
              const auto this_mass = (p1 + p2).M() / 1000.;
              if (std::abs(z_mass - this_mass) < std::abs(z_mass - best_mass)) {
                 best_mass = this_mass;
                 best_i1 = i1;
                 best_i2 = i2;
              }
           }
        }
    } else {
        for (size_t i = 0; i < idx_cmb[0].size(); i++) {
           const auto i1 = idx_cmb[0][i];
           const auto i2 = idx_cmb[1][i];
           if (charge[i1] * charge[i2] < 0) {
              ROOT::Math::PtEtaPhiEVector p1(pt[i1], eta[i1], phi[i1], e[i1]);
              ROOT::Math::PtEtaPhiEVector p2(pt[i2], eta[i2], phi[i2], e[i2]);
              const auto this_mass = (p1 + p2).M() / 1000.;
              if (std::abs(z_mass - this_mass) < std::abs(z_mass - best_mass)) {
                 best_mass = this_mass;
                 best_i1 = i1;
                 best_i2 = i2;
              }
           }
        }
    }
    
   idx[0].emplace_back(best_i1);
   idx[0].emplace_back(best_i2);
   if (charge[idx[0][0]] > 0.) {
        idx[0] = Reverse(idx[0]);
   }

   // Reconstruct virtual Z from remaining lepton pair
   for (size_t i = 0; i < 4; i++) {
      if (i != best_i1 && i != best_i2) {
         idx[1].emplace_back(i);
      }
   }
   if (charge[idx[1][0]] > 0.) {
        idx[1] = Reverse(idx[1]);
   }

   // Return indices of the pairs building two Z bosons
   return idx;
}

// Compute Z 4vectors from four leptons and the reconstructed pairs
std::vector<ROOT::Math::PtEtaPhiEVector> compute_z_vectors(const RVec<RVec<size_t>> &idx, rvec_f pt, rvec_f eta, rvec_f phi, rvec_f e)
{
   std::vector<ROOT::Math::PtEtaPhiEVector> z_vectors(2);
   for (size_t i = 0; i < 2; i++) {
      const auto i1 = idx[i][0]; const auto i2 = idx[i][1];
      ROOT::Math::PtEtaPhiEVector p1(pt[i1], eta[i1], phi[i1], e[i1]);
      ROOT::Math::PtEtaPhiEVector p2(pt[i2], eta[i2], phi[i2], e[i2]);
      z_vectors[i] = p1 + p2;
   }
   return z_vectors;
}

// Compute H 4vectors from the two reconstructed Z bosons
ROOT::Math::PtEtaPhiEVector compute_H_vector(std::vector<ROOT::Math::PtEtaPhiEVector> boson_vector)
{
   return boson_vector[0] + boson_vector[1];
}


//Order jets descending order pt, returning an ordered index
RVec<size_t> order_jets(rvec_f pt, int n)
{
    RVec<size_t> idx_j;
    idx_j.reserve(n);
    idx_j = Argsort(pt);
    return Reverse(idx_j);
}

//Return an 4-dim RVec of lepton indexes: 0-1 leading pair, 2-3 subleading; in each pair the first lep is with negative charge
RVec<size_t> get_idxlep(const RVec<RVec<size_t>> &idx)
{
    RVec<size_t> idx_lep;
    idx_lep.reserve(4);
    for (size_t i = 0; i < 2; i++) {
        const auto idx1 = idx[i][0]; const auto idx2 = idx[i][1];
        idx_lep.emplace_back(idx1); idx_lep.emplace_back(idx2);
    }
    return idx_lep;
}

// Compute lepton 4vectors in the Higgs c.o.m. from leptons variables: NOT ordered
std::vector<ROOT::Math::PtEtaPhiEVector> compute_lep_vectors_com_H(rvec_f pt, rvec_f eta, rvec_f phi, rvec_f e, ROOT::Math::PtEtaPhiEVector higgs_vector)
{
   std::vector<ROOT::Math::PtEtaPhiEVector> lep_vectors(4);
   for (size_t i = 0; i < 4; i++) {
      ROOT::Math::PtEtaPhiEVector p(pt[i], eta[i], phi[i], e[i]);
      lep_vectors[i] = ROOT::Math::VectorUtil::boost(p, higgs_vector.BoostToCM());
   }
   return lep_vectors;
}

// Compute the first 4-vector in the c.o.m. relative to the given second one
ROOT::Math::PtEtaPhiEVector com_vector(ROOT::Math::PtEtaPhiEVector daughter_vector, ROOT::Math::PtEtaPhiEVector mother_vector)
{
    return ROOT::Math::VectorUtil::boost(daughter_vector, mother_vector.BoostToCM());
}

// Compute the decay angle of the letpon w.r.t. its Z boson: the lep_vec is given in the Z boson c.o.m.
double compute_costheta12(ROOT::Math::PtEtaPhiEVector lep_vector, ROOT::Math::PtEtaPhiEVector z_vector)
{
    return ROOT::Math::VectorUtil::CosTheta(lep_vector.Vect(), z_vector.Vect());
}

// Compute the azimuthal angle between the decay planes of the Z bosons
double compute_phi(const ROOT::VecOps::RVec<ROOT::Math::PtEtaPhiEVector> &lep_vectors, const RVec<size_t> &idxlep, ROOT::Math::PtEtaPhiEVector z1_com_vector)
{
    ROOT::Math::DisplacementVector3D lead_planevector = lep_vectors[idxlep[0]].Vect().Cross(lep_vectors[idxlep[1]].Vect());
    ROOT::Math::DisplacementVector3D sub_planevector = lep_vectors[idxlep[2]].Vect().Cross(lep_vectors[idxlep[3]].Vect());
    if (lead_planevector.R() == 0 || sub_planevector.R() == 0) {
        std::cout << "Null cross product" << std::endl;
        return -10000;
    }
    ROOT::Math::DisplacementVector3D cross = lead_planevector.Cross(sub_planevector);
    double angle = ROOT::Math::VectorUtil::Angle(lead_planevector, sub_planevector);
    return (cross.Dot(z1_com_vector.Vect()) < 0) ? angle : -angle;
}

// Compute the azimuthal angle between the decay plane of the real Z boson and the plane of the real Z boson with the beam line
double compute_phi1(const ROOT::VecOps::RVec<ROOT::Math::PtEtaPhiEVector> &lep_vectors, const RVec<size_t> &idxlep, ROOT::Math::PtEtaPhiEVector z1_com_vector)
{
    ROOT::Math::DisplacementVector3D lead_planevector = lep_vectors[idxlep[0]].Vect().Cross(lep_vectors[idxlep[1]].Vect());
    ROOT::Math::Cartesian3D beam_axis(0, 0, 1);
    ROOT::Math::DisplacementVector3D Z1beam_planevector = z1_com_vector.Vect().Cross(beam_axis);
    if (lead_planevector.R() == 0 || Z1beam_planevector.R() == 0) {
        std::cout << "Null cross product" << std::endl;
        return -10000;
    }
    ROOT::Math::DisplacementVector3D cross = lead_planevector.Cross(Z1beam_planevector);
    double angle = ROOT::Math::VectorUtil::Angle(lead_planevector, Z1beam_planevector);
    return (cross.Dot(z1_com_vector.Vect()) < 0) ? angle : -angle;
}

//Adding the jet 4-vector (from jet variables) to the Higgs 4-vector (already passed as 4-vector)
ROOT::Math::PtEtaPhiEVector add_jet(ROOT::Math::PtEtaPhiEVector start_4vector, rvec_f pt, rvec_f eta, rvec_f phi, rvec_f e, size_t i)
{
    ROOT::Math::PtEtaPhiEVector p1(pt[i], eta[i], phi[i], e[i]);
   return start_4vector + p1;
}

//Compute the longitudinal Impact Parameter
double compute_z0sintheta(double pt, double eta, double phi, double e, double z0)
{
    ROOT::Math::PtEtaPhiEVector p(pt, eta, phi, e);
    return z0*sin(p.Theta());
}

//Compute the longitudinal Impact Parameter for an array of particles and return the long. IPs as RVec
RVec<Float_t> compute_z0sinthetavec(rvec_f pt, rvec_f eta, rvec_f phi, rvec_f e, rvec_f z0)
{
    RVec<Float_t> z0sintheta_vec;
    z0sintheta_vec.reserve(pt.size());
    for (size_t i = 0; i < pt.size(); i++) {
        ROOT::Math::PtEtaPhiEVector p(pt[i], eta[i], phi[i], e[i]);
        z0sintheta_vec.emplace_back(z0[i]*sin(p.Theta()));
    }
    return z0sintheta_vec;
}

//Given two azimuthal angle, return their difference in the [-pi,pi] range
double compute_deltaphi(double phi1, double phi2)
{
    double diff = phi1-phi2;
    if (diff < - TMath::Pi()) {
        diff += 2.0*TMath::Pi();
    } else if (diff > TMath::Pi()){
        diff -= 2.0*TMath::Pi();
    }
    return diff;
}

//====================================================================================

//Selecting collimated leptons from the same Z boson
bool filter_z_dr(const RVec<RVec<size_t>> &idx, rvec_f eta, rvec_f phi)
{
   for (size_t i = 0; i < 2; i++) {
      const auto i1 = idx[i][0];
      const auto i2 = idx[i][1];
      const auto dr = DeltaR(eta[i1], eta[i2], phi[i1], phi[i2]);
      if (dr < 0.01) {
         return false;
      }
   }
   return true;
};

#endif /*HIGGSFOURLEP_H */
