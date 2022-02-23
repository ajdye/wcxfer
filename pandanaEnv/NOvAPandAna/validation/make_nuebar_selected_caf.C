#include "CAFAna/Core/EventList.h"
#include "NDAna/nuebarcc_inc/NuebarCCIncCuts.h"
#include "NDAna/Classifiers/NueID.h"

#include "CAFAna/Core/Var.h"
#include <fstream>

#include <chrono>
#include <iostream>

std::vector<std::string> parse_file_list(std::string file_list)
{
  std::ifstream infile(file_list);
  std::vector<std::string> files;
  std::string this_file;
  while(infile >> this_file) {
    files.push_back(this_file);
  }
  return files;
}

void make_nuebar_selected_caf(std::string input_file_list)
{
  
  auto pos = input_file_list.find(".txt");
  auto output_file_name = input_file_list;
  output_file_name.replace(pos, 4, "_selected.root");

  auto kSelection = nuebarccinc::decaf::kPreselection;

  auto kRun    = SIMPLEVAR(hdr.run);
  auto kSubrun = SIMPLEVAR(hdr.subrun);
  auto kCycle  = SIMPLEVAR(hdr.cycle);
  auto kBatch  = SIMPLEVAR(hdr.batch);
  auto kEvent  = SIMPLEVAR(hdr.evt);
  auto kSlice  = SIMPLEVAR(hdr.subevt);

  std::vector<std::pair<std::string, ana::Var> > int_vars = {
    {"run", kRun},
    {"subrun", kSubrun},
    {"cycle", kCycle},
    {"batch", kBatch},
    {"evt", kEvent},
    {"subevt", kSlice},
  };

  std::vector<std::pair<std::string, ana::Var> > float_vars = {
    {"NueID", ana::nueid_classifier::kNueID},
    {"electronid", ana::nueid_classifier::kElectronID},
    {"epi0llt", ana::nueid_classifier::kEPi0LLT},
    {"width", ana::nueid_classifier::kShwWidth},
    {"gap", ana::nueid_classifier::kShwGap},
  };

  auto start = std::chrono::high_resolution_clock::now();
  ana::MakeEventTTreeFile(parse_file_list(input_file_list),
			  output_file_name,
			  {{"selection", kSelection}},
			  float_vars,
			  int_vars);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  std::cout << duration.count() << " ns" << std::endl;

}
