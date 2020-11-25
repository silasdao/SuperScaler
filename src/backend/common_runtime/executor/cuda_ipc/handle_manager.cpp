// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "handle_manager.hpp"
#include "cuda_ipc_internal.hpp"

HandleManager::HandleManager()
{
}

HandleManager::~HandleManager()
{
    for (auto &pair : m_handle_cache) {
        DeviceContextGuard guard(pair.second.dev_id);
        checkCudaErrors(cudaIpcCloseMemHandle(pair.second.dev_ptr));
    }
}

void *HandleManager::get_address(const cudaIpcMemHandle_t &handle, int dev_id)
{
    auto itr = m_handle_cache.find(handle);
    if (itr == m_handle_cache.end()) {
        void *buffer = nullptr;
        DeviceContextGuard guard(dev_id);
        checkCudaErrors(cudaIpcOpenMemHandle(&buffer, handle, cudaIpcMemLazyEnablePeerAccess));
        m_handle_cache.emplace(handle, HandleInfo{buffer, dev_id});
        return buffer;
    } else {
        return itr->second.dev_ptr;
    }
}

bool HandleManager::free_address(const cudaIpcMemHandle_t &handle)
{
    auto itr = m_handle_cache.find(handle);
    if (itr == m_handle_cache.end())
        return false;
    else {
        DeviceContextGuard guard(itr->second.dev_id);
        checkCudaErrors(cudaIpcCloseMemHandle(itr->second.dev_ptr));
        m_handle_cache.erase(itr);
        return true;
    }
}
