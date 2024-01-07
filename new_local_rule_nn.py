
    def get_random_matrix_inds(self, param):
        num_update_elements = int(param.nelement()*self.prop_cells_updated)
        random_inds = torch.randperm(param.nelement())[:num_update_elements]
        row_indices = random_inds // param.size(1)
        col_indices = random_inds % param.size(1)
        index_tuples = list(zip(row_indices.tolist(), col_indices.tolist()))
        return index_tuples

    def get_forward_states(self, param, hidden_states, i, j):
        # given the parameter index, what are the neighbors?
        if i == 0: # if the back unit is the first one
            forward = param[i+1:, j].flatten() # index all forward weights besides the
                                               # current one
            forward_states = hidden_states[i+1:,j,:]
        elif i == param.shape[0]: # if the back unit is the last one
            forward = param[:i, j].flatten() # index all except the last one
            forward_states = hidden_states[:i,j,:] 
        else: # and if it's in the middle, concat the everything before and everything after
            forward = torch.cat((param[:i,j].flatten(), param[i+1:, j].flatten())) # 
            forward_states = torch.cat((hidden_states[:i,j], hidden_states[i+1:, j]))
        return forward, forward_states

    def get_backward_states(self, param, hidden_states, i, j):
        if j == 0:
            backward = param[i, j+1:].flatten()
            backward_states = hidden_states[i,j+1:,:]
        elif j == param.shape[1]:
            backward = param[i, :j].flatten()
            backward_states = hidden_states[i,:j,:]
        else:
            backward = torch.cat((param[i,:j].flatten(), param[i, j+1:].flatten()))
            #backward_states = torch.cat((hidden_states[i,:j], hidden_states[i, j+1:]), dim=1)
            #concatenate backward states to average later
            backward_states = torch.cat((hidden_states[i,:j], hidden_states[i, j+1:]), dim=0)
        return backward, backward_states

    def get_neighboring_signals(self, param, hidden_states, i, j):
        forward, forward_states = self.get_forward_states(param, hidden_states, i, j)
        backward, backward_states = self.get_backward_states(param, hidden_states, i, j)

        concatted_weights = torch.cat((param[i,j][None], torch.mean(forward)[None], torch.mean(backward)[None]))
        concatted_states = torch.cat((hidden_states[i,j, :].flatten(), torch.mean(forward_states,
            dim=0).flatten(), torch.mean(backward_states, dim=0).flatten()))

        concatted = torch.cat((concatted_weights, concatted_states))
        return concatted

    def nca_local_rule(self, param, hidden_states): 
        torch.autograd.set_detect_anomaly(True)
        update_index_tuples = self.get_random_matrix_inds(param)
        all_perceptions = torch.zeros(len(update_index_tuples), 3 + 3*self.hidden_state_dim).to(self.device)
        # in order to get forward weights and forward_states, i want to mask
        for k, (i,j) in enumerate(update_index_tuples):
            all_perceptions[k, :] = self.get_neighboring_signals(param, hidden_states, i, j)

        batchwise_updates = self.local_nn(all_perceptions) # calculate all updates at once
        reshaped_updates = torch.zeros(param.shape[0], param.shape[1], 1 +
            self.hidden_state_dim).to(self.device)

        for k, (i,j) in enumerate(update_index_tuples):
            reshaped_updates[i, j, :] = batchwise_updates[k, :]

        weight_updates = reshaped_updates[:,:,0] 
        hidden_state_updates = reshaped_updates[:,:,1:]
        return weight_updates, hidden_state_updates
