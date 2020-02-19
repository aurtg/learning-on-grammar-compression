// Copyright (C) 2020 NEC Corporation
// See LICENSE.

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <queue>
#include <set>
#include <map>
#include <utility>

using namespace::std;

// depth: length of the list is equal to #node. all elements are initialized by 0
void _forward_depth(vector<int> &seed_node, vector<int> &depth,
    vector<set<int> > &in_edge_list, vector<set<int> > &out_edge_list){
    //Initialization
    queue<pair<int, int> > edge_queue;
    for(int i: seed_node){
        for(int o: out_edge_list[i]){
            pair<int, int> edge = {i, o};
            edge_queue.push(edge);
        }
    }

    //Calculate depth for each node.
    while(edge_queue.size() > 0){
        pair<int, int> edge = edge_queue.front();
        edge_queue.pop();
        int i = edge.first;
        int o = edge.second;

        in_edge_list[o].erase(i);
        if(in_edge_list[o].size()==0){
            for(int oo: out_edge_list[o]){
                pair<int, int> new_edge = {o, oo};
                edge_queue.push(new_edge);
            }

            depth[o] = max({depth[o], depth[i]+1});
        }
    }
}

int _backward_depth(vector<int> &leaf_node, vector<int> &depth,
  vector<set<int> > &in_edge_list, vector<set<int> > &out_edge_list){
    // Initialization
    int cur_depth = 0;
    queue<int> node_queue;
    for(int i: leaf_node){
      node_queue.push(i);
    }

    // Calculate depth for each node.
    while(node_queue.size() > 0){
        int node = node_queue.front();
        node_queue.pop();

        // Currently processing depth.
        if(depth[node] < cur_depth){
            cur_depth = depth[node];
        }

        // For all parent node:
        for(int parent: in_edge_list[node]){
            out_edge_list[parent].erase(node);

            // If parent node is in next depth
            if(out_edge_list[parent].size()==0){
                // Root node with no in-edge is ignored.
                if(in_edge_list[parent].size()>0){
                    depth[parent] = cur_depth - 1;

                    node_queue.push(parent);
                }
            }
        }
    }

    return cur_depth;
}

inline void _convert_vector_to_list(vector<vector<int> > &inp, vector<set<int> > &out){
    for(vector<int> nodes: inp){
      set<int> new_nodes;
      for(int node: nodes){
        new_nodes.insert(node);
      }
      out.push_back(new_nodes);
    }
}

class Groups{
public:
    vector<vector<int> > char_group;
    int max_depth;
    int length_dummy_seq; // Only used in re-pair encoder.
    vector<vector<vector<int> > > groups;

    vector<vector<int> > non_char_segments;

    Groups(){
      for(int i=0; i<3; i++){
        vector<int> tmp;
        this->char_group.push_back(tmp);
      }

      this->max_depth=0;
    }

    void append_new_group(){
        this->max_depth ++;
        vector<vector<int> > new_group;
        for(int i=0; i<4; i++){
            vector<int> tmp;
            new_group.push_back(tmp);
        }
        this->groups.push_back(new_group);
    }
};

// Return: <character> id_seq, id_pos, val, <depth1> first_id_seq, first_id_pos,
//    second_id_seq, second_id_pos, to_id_seq, to_id_pos, <depth2> ... <depthN>
Groups create_computation_groups(vector<vector<vector<int> > > lzd_seqs, int mode){
    Groups groups;

    for(int i_seq=0; i_seq < lzd_seqs.size(); i_seq++){
        vector<vector<int> > lzd_seq = lzd_seqs[i_seq];

        vector<int> seed_node;
        vector<int> depths(lzd_seq.size(), 0);
        vector<set<int> > in_edge_list;
        vector<set<int> > out_edge_list;
        for(int i_pos=0; i_pos<lzd_seq.size(); i_pos++){
            set<int> in_edges;
            set<int> out_edges;
            in_edge_list.push_back(in_edges);
            out_edge_list.push_back(out_edges);
        }

        // Create edge list.
        for(int i_pos=0; i_pos<lzd_seq.size(); i_pos++){
            // Segment of single character.
            if(lzd_seq[i_pos][0]==0){
                seed_node.push_back(i_pos);
            }
            // Segment with composition.
            else{
                // Add 2 edges; component -> this segment
                in_edge_list[i_pos].insert(lzd_seq[i_pos][0]);
                in_edge_list[i_pos].insert(lzd_seq[i_pos][1]);

                out_edge_list[lzd_seq[i_pos][0]].insert(i_pos);
                out_edge_list[lzd_seq[i_pos][1]].insert(i_pos);
            }
        }

        // Forward grouping
        if (mode==0){
            // depth
            _forward_depth(seed_node, depths, in_edge_list, out_edge_list);

            // register nodes to groups
            vector<int> pos_non_char;
            for(int i_pos=0; i_pos<lzd_seq.size(); i_pos++){
                if(lzd_seq[i_pos][0]==0){
                    groups.char_group[0].push_back(i_seq);
                    groups.char_group[1].push_back(i_pos);
                    groups.char_group[2].push_back(lzd_seq[i_pos][1]);
                }
                else{
                    while(groups.max_depth < depths[i_pos]){
                        groups.append_new_group();
                    }

                    int depth = depths[i_pos] - 1;
                    groups.groups[depth][0].push_back(i_seq);
                    groups.groups[depth][1].push_back(lzd_seq[i_pos][0]);
                    groups.groups[depth][2].push_back(lzd_seq[i_pos][1]);
                    groups.groups[depth][3].push_back(i_pos);

                    pos_non_char.push_back(i_pos);
                }
            }
            groups.non_char_segments.push_back(pos_non_char);
        }
        // Backward grouping
        else if (mode==1){
            vector<int> leaf_node;
            for(int i_pos=0; i_pos<lzd_seq.size(); i_pos++){
                if(out_edge_list[i_pos].size() == 0){
                    leaf_node.push_back(i_pos);
                }
            }

            int min_depth = _backward_depth(leaf_node, depths, in_edge_list, out_edge_list);

            // register nodes to groups
            vector<int> pos_non_char;
            for(int i_pos=0; i_pos<lzd_seq.size(); i_pos++){
                if(lzd_seq[i_pos][0]==0){
                    groups.char_group[0].push_back(i_seq);
                    groups.char_group[1].push_back(i_pos);
                    groups.char_group[2].push_back(lzd_seq[i_pos][1]);
                }
                else{
                    int true_depth = depths[i_pos] - min_depth + 1;

                    while(groups.max_depth < true_depth){
                        groups.append_new_group();
                    }

                    int depth = true_depth - 1;
                    groups.groups[depth][0].push_back(i_seq);
                    groups.groups[depth][1].push_back(lzd_seq[i_pos][0]);
                    groups.groups[depth][2].push_back(lzd_seq[i_pos][1]);
                    groups.groups[depth][3].push_back(i_pos);

                    pos_non_char.push_back(i_pos);
                }
            }
            groups.non_char_segments.push_back(pos_non_char);
        }
    }

    return groups;
}

inline void _prepare_blank_vectors(vector<set<int> > &input, int insert_idx){
  while(input.size() < insert_idx + 1){
      set<int> tmp;
      input.push_back(tmp);
  }
}

Groups create_computation_groups_repair(vector<pair<vector<vector<int> >, vector<int> > > seqs, int mode){
    Groups groups;

    groups.length_dummy_seq = 0;

    for(int i_seq=0; i_seq < seqs.size(); i_seq++){
        vector<vector<int> > &rules = seqs[i_seq].first;
        vector<int> &comp_seq = seqs[i_seq].second;

        vector<int> seed_node;
        vector<set<int> > in_edge_list;
        vector<set<int> > out_edge_list;

        // For padding index (0)
        seed_node.push_back(0);

        // Index which appears in original compressed sequence -> Index of dummy factor.
        map<int, int> factor_position;

        // Dummy factor -> pair of dummy factors of its parent
        map<int, pair<int, int> > reverse_rule;

        // Position index in compressed sequence -> index of corresponding dummy factor.
        vector<int> factor_position_seq;

        // Load rules.
        int factor_id = 256;
        for (vector<int> rule : rules){
            // Register new factor or character.
            for(int ph: rule){
                // If ph is a new character.
                if(ph < 256){
                    if(factor_position.find(ph) == factor_position.end()){
                        int pos = factor_position.size() + 1;
                        factor_position[ph] = pos;

                        seed_node.push_back(factor_position[ph]);

                        groups.char_group[0].push_back(i_seq);
                        groups.char_group[1].push_back(factor_position[ph]);
                        groups.char_group[2].push_back(ph);
                    }
                }
            }
            int pos = factor_position.size() + 1;
            factor_position[factor_id] = pos;

            // Register edges which represent this rule.
            int child = factor_position[factor_id];
            for(int ph: rule){
                int parent = factor_position[ph];
                _prepare_blank_vectors(in_edge_list, child);
                in_edge_list[child].insert(parent);
                _prepare_blank_vectors(out_edge_list, parent);
                out_edge_list[parent].insert(child);
            }

            reverse_rule[factor_position[factor_id]] = \
                {factor_position[rule[0]], factor_position[rule[1]]};

            factor_id++;
        }

        // Load sequences
        for(int token: comp_seq){
            // If token is unknown character, register it.
            if(token < 256){
                if(factor_position.find(token) == factor_position.end()){
                    int pos = factor_position.size() + 1;
                    factor_position[token] = pos;

                    seed_node.push_back(factor_position[token]);

                    groups.char_group[0].push_back(i_seq);
                    groups.char_group[1].push_back(factor_position[token]);
                    groups.char_group[2].push_back(token);
                }
            }

            factor_position_seq.push_back(factor_position[token]);
        }

        _prepare_blank_vectors(in_edge_list, factor_position.size());
        _prepare_blank_vectors(out_edge_list, factor_position.size());

        if (groups.length_dummy_seq < factor_position.size() + 1){
            groups.length_dummy_seq = factor_position.size() + 1;
        }

        // Forward grouping
        if (mode==0){
            //depth
            vector<int> depths(factor_position.size()+1, 0);
            _forward_depth(seed_node, depths, in_edge_list, out_edge_list);

            // register nodes to groups
            for(auto key_value: factor_position){
                int code = key_value.first;
                int dummy_factor = key_value.second;

                // Character has been already registered to char_group.
                if(code < 256){
                    continue;
                }

                while(groups.max_depth < depths[dummy_factor]){
                    groups.append_new_group();
                }

                int depth = depths[dummy_factor] - 1;

                groups.groups[depth][0].push_back(i_seq);
                groups.groups[depth][1].push_back(reverse_rule[dummy_factor].first);
                groups.groups[depth][2].push_back(reverse_rule[dummy_factor].second);
                groups.groups[depth][3].push_back(dummy_factor);
            }
        }
        else if(mode==1){
            vector<int> depths(factor_position.size()+1, 0);

            vector<int> leaf_node;
            for(int i_pos=0; i_pos<factor_position.size()+1; i_pos++){
                if(out_edge_list[i_pos].size() == 0){
                    leaf_node.push_back(i_pos);
                }
            }

            int min_depth = _backward_depth(leaf_node, depths, in_edge_list,
                out_edge_list);

            // register nodes to groups
            for(auto key_value: factor_position){
                int code = key_value.first;
                int dummy_factor = key_value.second;

                // Character has been already registered to char_group.
                if(code < 256){
                    continue;
                }

                int true_depth = depths[dummy_factor] - min_depth + 1;

                while(groups.max_depth < true_depth){
                    groups.append_new_group();
                }

                int depth = true_depth - 1;
                groups.groups[depth][0].push_back(i_seq);
                groups.groups[depth][1].push_back(reverse_rule[dummy_factor].first);
                groups.groups[depth][2].push_back(reverse_rule[dummy_factor].second);
                groups.groups[depth][3].push_back(dummy_factor);
            }
        }

        groups.non_char_segments.push_back(factor_position_seq);
    }

    return groups;
}

//Wrapper functions for debugging
vector<int> forward_depth(int n_node, vector<int> &seed_node,
    vector<vector<int> > &in_edge_list, vector<vector<int> > &out_edge_list){
    vector<int> depth(n_node, 0);
    vector<set<int> > new_in_edge_list;
    vector<set<int> > new_out_edge_list;
    _convert_vector_to_list(in_edge_list, new_in_edge_list);
    _convert_vector_to_list(out_edge_list, new_out_edge_list);

    _forward_depth(seed_node, depth, new_in_edge_list, new_out_edge_list);

    return depth;
}

vector<int> backward_depth(int n_node, vector<int> &leaf_node,
    vector<vector<int> > &in_edge_list, vector<vector<int> > &out_edge_list){
    vector<int> depth(n_node, 0);
    vector<set<int> > new_in_edge_list;
    vector<set<int> > new_out_edge_list;
    _convert_vector_to_list(in_edge_list, new_in_edge_list);
    _convert_vector_to_list(out_edge_list, new_out_edge_list);

    _backward_depth(leaf_node, depth, new_in_edge_list, new_out_edge_list);

    return depth;
}

namespace py = pybind11;
PYBIND11_PLUGIN(grouping){
    py::module m("grouping", "grouping");

    m.def("forward_depth", &forward_depth);
    m.def("backward_depth", &backward_depth);

    py::class_<Groups>(m, "Groups")
      .def_readwrite("char_group", &Groups::char_group)
      .def_readwrite("groups", &Groups::groups)
      .def_readwrite("non_char_segments", &Groups::non_char_segments)
      .def_readwrite("length_dummy_seq", &Groups::length_dummy_seq);

    m.def("create_computation_groups", &create_computation_groups);
    m.def("create_computation_groups_repair", &create_computation_groups_repair);

    return m.ptr();
}
