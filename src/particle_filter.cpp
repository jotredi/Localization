/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;

// Function for Multivariate Gaussian distribution
inline double multi_gauss(double x, double y,
                          double mu_x, double mu_y,
                          double sigma_x, double sigma_y){
  // Normalization
  double norm = 1.0/(2*M_PI*sigma_x*sigma_y);
  // Exponent
  double exponent = (pow(x-mu_x,2)/(2*pow(sigma_x,2))) + (pow(y-mu_y,2)/(2*pow(sigma_y,2)));

  return norm * exp(-exponent);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Set number of particles
  num_particles = 100;

  // Set particles & weights vector with num_particles
  particles.resize(num_particles);
  weights.resize(num_particles, 1.0);

  // Random engine
  std::default_random_engine gen;

  // Normal distributions for x, y & theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize each particle
  for(int i=0; i<num_particles; ++i){
    // Set id
    particles[i].id = i + 1;
    // Sample x
    particles[i].x = dist_x(gen);
    // Sample y
    particles[i].y = dist_y(gen);
    // Sample theta
    particles[i].theta = dist_theta(gen);
    // Set weight
    particles[i].weight = 1.0;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

  // Random engine
  std::default_random_engine gen;

  // Normal distributions for x, y & theta
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

  // Check if yaw rate is not zero
  if (fabs(yaw_rate) > 0.0001){
    // Add measurements to each particle
    for (int i=0; i<num_particles; ++i){

      // Last position & orientation
      double x0 = particles[i].x;
      double y0 = particles[i].y;
      double theta0 = particles[i].theta;

      // Calculate new position & orientation
      double xf = x0 + (velocity/yaw_rate)*(sin(theta0 + yaw_rate*delta_t) - sin(theta0));
      double yf = y0 + (velocity/yaw_rate)*(cos(theta0) - cos(theta0 + yaw_rate*delta_t));
      double thetaf = theta0 + yaw_rate*delta_t;

      // Set new ones with additive Gaussian noise
      particles[i].x = xf + dist_x(gen);
      particles[i].y = yf + dist_y(gen);
      particles[i].theta = thetaf + dist_theta(gen);
    }
  }
  else{
    // if yaw rate = 0
    for (int i=0; i<num_particles; ++i){

      // Last position & orientation
      double x0 = particles[i].x;
      double y0 = particles[i].y;
      double theta0 = particles[i].theta;

      // Calculate new position & orientation
      double xf = x0 + velocity*cos(theta0)*delta_t;
      double yf = y0 + velocity*sin(theta0)*delta_t;
      double thetaf = theta0;

      // Set new ones with additive Gaussian noise
      particles[i].x = xf + dist_x(gen);
      particles[i].y = yf + dist_y(gen);
      particles[i].theta = thetaf + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {

  // Assign each observed measurement with the nearest landmark id
  for(LandmarkObs& obs : observations){
    double min_dist = std::numeric_limits<double>::infinity();
    int min_id;

    // For each predicted observation
    for(const LandmarkObs& pred : predicted){
      // Calculate distance
      double distance = dist(obs.x, obs.y, pred.x, pred.y);

      if(distance < min_dist){
        min_dist = distance;
        min_id = pred.id;
      }
    }
    // Set associated landmark id
    obs.id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  // Landmark measurement uncertainty
  double sigma_x = std_landmark[0], sigma_y = std_landmark[1];

  // Update weights of each particle
  for (int i=0; i<num_particles; ++i){

    // Get particle pos & orientation
    double xp = particles[i].x;
    double yp = particles[i].y;
    double theta = particles[i].theta;

    // Predict measurements to map landmarks
    vector<LandmarkObs> predicted;

    // for each landmark in the map
    for(const auto& landmark : map_landmarks.landmark_list){
      // Calculate distance from particle pos to landmark
      double distance = dist(xp, yp, landmark.x_f, landmark.y_f);

      // Append to predicted measurements if within sensor range
      if(distance < sensor_range){
        LandmarkObs pred;
        pred.id = landmark.id_i;
        pred.x = landmark.x_f;
        pred.y = landmark.y_f;
        predicted.push_back(pred);
      }
    }

    // Transformation from car to map coordinate frame
    vector<LandmarkObs> map_observations;

    // for each observation
    for(const LandmarkObs& obs : observations){
      LandmarkObs map_obs;
      map_obs.x = obs.x*cos(theta) - obs.y*sin(theta) + xp;
      map_obs.y = obs.x*sin(theta) + obs.y*cos(theta) + yp;
      map_observations.push_back(map_obs);
    }

    // Associate transformed observations with landmark id's
    dataAssociation(predicted, map_observations);

    // Calculate new particle's weight
    weights[i] = 1.0;

    // Delete last associations
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();

    // for each observation
    for (const LandmarkObs& obs : map_observations){
      // Get id and x,y associations
      int map_id = obs.id;
      double x = obs.x;
      double y = obs.y;

      // Set associations
      particles[i].associations.push_back(map_id);
      particles[i].sense_x.push_back(x);
      particles[i].sense_y.push_back(y);

      // Calculate mean of Multivariate-Gaussian (associated landmark position)
      double mu_x = map_landmarks.landmark_list[map_id-1].x_f;
      double mu_y = map_landmarks.landmark_list[map_id-1].y_f;

      // Evaluate on x,y measurement position
      double prob = multi_gauss(x, y, mu_x, mu_y, sigma_x, sigma_y);

      // Update weight
      weights[i] *= prob;
    }

    // Set particle weight
    particles[i].weight = weights[i];
  }
}

void ParticleFilter::resample() {

  // Random engine
  std::default_random_engine gen;
  // Particle discrete distribution
  std::discrete_distribution<int> part_dist (weights.begin(), weights.end());

  // Resample each particle
  vector<Particle> new_particles;

  for(int i=0; i<num_particles; ++i){
    new_particles.push_back(particles[part_dist(gen)]);
  }

  // Set new particles
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
