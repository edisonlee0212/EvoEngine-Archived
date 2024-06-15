#include "Prefab.hpp"
#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "ProjectManager.hpp"
#include "Serialization.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "TransformGraph.hpp"
#include "UnknownPrivateComponent.hpp"
#include "Utilities.hpp"
using namespace evo_engine;
void Prefab::OnCreate() {
  instance_name = "New Prefab";
}

#pragma region Assimp Import
struct AssimpImportNode {
  aiNode* m_correspondingNode = nullptr;
  std::string m_name;
  Transform m_localTransform;
  AssimpImportNode(aiNode* node);
  std::shared_ptr<AssimpImportNode> m_parent;
  std::vector<std::shared_ptr<AssimpImportNode>> m_children;
  std::shared_ptr<Bone> m_bone;
  bool m_hasMesh;

  bool NecessaryWalker(std::unordered_map<std::string, std::shared_ptr<Bone>>& boneMap);
  void AttachToAnimator(const std::shared_ptr<Animation>& animation, size_t& index) const;
  void AttachChild(const std::shared_ptr<Bone>& parent, size_t& index) const;
};
glm::mat4 mat4_cast(const aiMatrix4x4& m) {
  return glm::transpose(glm::make_mat4(&m.a1));
}
aiMatrix4x4 mat4_cast(const glm::mat4& m) {
  aiMatrix4x4 retVal;
  retVal.a1 = m[0][0];
  retVal.a2 = m[1][0];
  retVal.a3 = m[2][0];
  retVal.a4 = m[3][0];

  retVal.b1 = m[0][1];
  retVal.b2 = m[1][1];
  retVal.b3 = m[2][1];
  retVal.b4 = m[3][1];

  retVal.c1 = m[0][2];
  retVal.c2 = m[1][2];
  retVal.c3 = m[2][2];
  retVal.c4 = m[3][2];

  retVal.d1 = m[0][3];
  retVal.d2 = m[1][3];
  retVal.d3 = m[2][3];
  retVal.d4 = m[3][3];
  return retVal;
}
glm::mat4 mat4_cast(const aiMatrix3x3& m) {
  return glm::transpose(glm::make_mat3(&m.a1));
}
AssimpImportNode::AssimpImportNode(aiNode* node) {
  m_correspondingNode = node;
  if (node->mParent)
    m_localTransform.value = mat4_cast(node->mTransformation);
  m_name = node->mName.C_Str();
}
void AssimpImportNode::AttachToAnimator(const std::shared_ptr<Animation>& animation, size_t& index) const {
  animation->root_bone = m_bone;
  animation->root_bone->index = index;
  for (auto& i : m_children) {
    index += 1;
    i->AttachChild(m_bone, index);
  }
}
void AssimpImportNode::AttachChild(const std::shared_ptr<Bone>& parent, size_t& index) const {
  m_bone->index = index;
  parent->m_children.push_back(m_bone);
  for (auto& i : m_children) {
    index += 1;
    i->AttachChild(m_bone, index);
  }
}
bool AssimpImportNode::NecessaryWalker(std::unordered_map<std::string, std::shared_ptr<Bone>>& boneMap) {
  bool necessary = false;
  for (int i = 0; i < m_children.size(); i++) {
    if (!m_children[i]->NecessaryWalker(boneMap)) {
      m_children.erase(m_children.begin() + i);
      i--;
    } else {
      necessary = true;
    }
  }
  const auto search = boneMap.find(m_name);
  if (search != boneMap.end()) {
    m_bone = search->second;
    necessary = true;
  } else if (necessary) {
    m_bone = std::make_shared<Bone>();
    m_bone->name = m_name;
  }

  return necessary;
}
void ReadKeyFrame(BoneKeyFrames& boneAnimation, const aiNodeAnim* channel) {
  const auto numPositions = channel->mNumPositionKeys;
  boneAnimation.positions.resize(numPositions);
  for (int positionIndex = 0; positionIndex < numPositions; ++positionIndex) {
    const aiVector3D aiPosition = channel->mPositionKeys[positionIndex].mValue;
    const float timeStamp = channel->mPositionKeys[positionIndex].mTime;
    BonePosition data;
    data.value = glm::vec3(aiPosition.x, aiPosition.y, aiPosition.z);
    data.time_stamp = timeStamp;
    boneAnimation.positions.push_back(data);
    boneAnimation.max_time_stamp = glm::max(boneAnimation.max_time_stamp, timeStamp);
  }

  const auto numRotations = channel->mNumRotationKeys;
  boneAnimation.rotations.resize(numRotations);
  for (int rotationIndex = 0; rotationIndex < numRotations; ++rotationIndex) {
    const aiQuaternion aiOrientation = channel->mRotationKeys[rotationIndex].mValue;
    const float timeStamp = channel->mRotationKeys[rotationIndex].mTime;
    BoneRotation data;
    data.value = glm::quat(aiOrientation.w, aiOrientation.x, aiOrientation.y, aiOrientation.z);
    data.time_stamp = timeStamp;
    boneAnimation.rotations.push_back(data);
    boneAnimation.max_time_stamp = glm::max(boneAnimation.max_time_stamp, timeStamp);
  }

  const auto numScales = channel->mNumScalingKeys;
  boneAnimation.scales.resize(numScales);
  for (int keyIndex = 0; keyIndex < numScales; ++keyIndex) {
    const aiVector3D scale = channel->mScalingKeys[keyIndex].mValue;
    const float timeStamp = channel->mScalingKeys[keyIndex].mTime;
    BoneScale data;
    data.m_value = glm::vec3(scale.x, scale.y, scale.z);
    data.time_stamp = timeStamp;
    boneAnimation.scales.push_back(data);
    boneAnimation.max_time_stamp = glm::max(boneAnimation.max_time_stamp, timeStamp);
  }
}
void ReadAnimations(const aiScene* importerScene, const std::shared_ptr<Animation>& animator,
                    std::unordered_map<std::string, std::shared_ptr<Bone>>& bonesMap) {
  for (int i = 0; i < importerScene->mNumAnimations; i++) {
    aiAnimation* importerAnimation = importerScene->mAnimations[i];
    const std::string animationName = importerAnimation->mName.C_Str();
    float maxAnimationTimeStamp = 0.0f;
    for (int j = 0; j < importerAnimation->mNumChannels; j++) {
      const aiNodeAnim* importerNodeAnimation = importerAnimation->mChannels[j];
      const std::string nodeName = importerNodeAnimation->mNodeName.C_Str();
      const auto search = bonesMap.find(nodeName);
      if (search != bonesMap.end()) {
        auto& bone = search->second;
        bone->animations[animationName] = BoneKeyFrames();
        ReadKeyFrame(bone->animations[animationName], importerNodeAnimation);
        maxAnimationTimeStamp = glm::max(maxAnimationTimeStamp, bone->animations[animationName].max_time_stamp);
      }
    }
    animator->animation_length[animationName] = maxAnimationTimeStamp;
  }
}
std::shared_ptr<Texture2D> CollectTexture(const std::string& directory, const std::string& path,
                                          std::unordered_map<std::string, std::shared_ptr<Texture2D>>& loadedTextures) {
  const auto fullPathStr = directory + "\\" + path;
  std::string fullPath = std::filesystem::absolute(std::filesystem::path(fullPathStr)).string();
  if (!std::filesystem::exists(fullPath)) {
    fullPath = std::filesystem::absolute(path).string();
  }
  if (!std::filesystem::exists(fullPath)) {
    fullPath = std::filesystem::absolute(directory + "\\" + std::filesystem::path(path).filename().string()).string();
  }

  if (!std::filesystem::exists(fullPath)) {
    auto baseDir = std::filesystem::absolute(directory);
    fullPath =
        std::filesystem::absolute(baseDir.parent_path() / std::filesystem::path(path).filename().string()).string();
  }
  if (!std::filesystem::exists(fullPath)) {
    auto baseDir = std::filesystem::absolute(directory);
    fullPath = std::filesystem::absolute(baseDir.parent_path().parent_path() / "textures" /
                                         std::filesystem::path(path).filename().string())
                   .string();
  }
  if (!std::filesystem::exists(fullPath)) {
    auto baseDir = std::filesystem::absolute(directory);
    fullPath = std::filesystem::absolute(baseDir.parent_path().parent_path() / "texture" /
                                         std::filesystem::path(path).filename().string())
                   .string();
  }
  if (!std::filesystem::exists(fullPath)) {
    return Resources::GetResource<Texture2D>("TEXTURE_MISSING");
  }
  if (const auto search = loadedTextures.find(fullPath); search != loadedTextures.end()) {
    return search->second;
  }
  std::shared_ptr<Texture2D> texture2D;
  if (ProjectManager::IsInProjectFolder(fullPath)) {
    texture2D = std::dynamic_pointer_cast<Texture2D>(
        ProjectManager::GetOrCreateAsset(ProjectManager::GetPathRelativeToProject(fullPath)));
  } else {
    texture2D = ProjectManager::CreateTemporaryAsset<Texture2D>();
    texture2D->Import(fullPath);
  }
  loadedTextures[fullPath] = texture2D;
  return texture2D;
}
std::shared_ptr<Material> ReadMaterial(
    const std::string& directory, std::unordered_map<std::string, std::shared_ptr<Texture2D>>& loadedTextures,
    std::vector<std::pair<std::shared_ptr<Texture2D>, std::shared_ptr<Texture2D>>>& opacityMaps,
    const aiMaterial* importerMaterial) {
  auto targetMaterial = ProjectManager::CreateTemporaryAsset<Material>();
  if (importerMaterial) {
    // PBR
    if (importerMaterial->GetTextureCount(aiTextureType_BASE_COLOR) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_BASE_COLOR, 0, &str);
      targetMaterial->SetAlbedoTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }
    if (importerMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &str);
      targetMaterial->SetAlbedoTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }
    if (importerMaterial->GetTextureCount(aiTextureType_NORMALS) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_NORMALS, 0, &str);
      targetMaterial->SetNormalTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    } else if (importerMaterial->GetTextureCount(aiTextureType_HEIGHT) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_HEIGHT, 0, &str);
      targetMaterial->SetNormalTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    } else if (importerMaterial->GetTextureCount(aiTextureType_NORMAL_CAMERA) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_NORMAL_CAMERA, 0, &str);
      targetMaterial->SetNormalTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }

    if (importerMaterial->GetTextureCount(aiTextureType_METALNESS) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_METALNESS, 0, &str);
      targetMaterial->SetMetallicTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }
    if (importerMaterial->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &str);
      targetMaterial->SetRoughnessTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }
    if (importerMaterial->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &str);
      targetMaterial->SetAOTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }
    if (importerMaterial->GetTextureCount(aiTextureType_OPACITY) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_OPACITY, 0, &str);
      const auto opacityTexture = CollectTexture(directory, str.C_Str(), loadedTextures);
      const auto albedoTexture = targetMaterial->GetAlbedoTexture();

      opacityMaps.emplace_back(albedoTexture, opacityTexture);
    }
    if (importerMaterial->GetTextureCount(aiTextureType_TRANSMISSION) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_TRANSMISSION, 0, &str);
      const auto opacityTexture = CollectTexture(directory, str.C_Str(), loadedTextures);
      const auto albedoTexture = targetMaterial->GetAlbedoTexture();

      opacityMaps.emplace_back(albedoTexture, opacityTexture);
    }
    /*
    if (importerMaterial->GetTextureCount(aiTextureType_LIGHTMAP) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_LIGHTMAP, 0, &str);
      targetMaterial->SetAOTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }
    if (importerMaterial->GetTextureCount(aiTextureType_AMBIENT) > 0) {
      aiString str;
      importerMaterial->GetTexture(aiTextureType_AMBIENT, 0, &str);
      targetMaterial->SetAOTexture(CollectTexture(directory, str.C_Str(), loadedTextures));
    }*/
    int unknownTextureSize = 0;
    if (importerMaterial->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
      unknownTextureSize++;
    }
    if (importerMaterial->GetTextureCount(aiTextureType_SHININESS) > 0) {
      unknownTextureSize++;
    }
    if (importerMaterial->GetTextureCount(aiTextureType_DISPLACEMENT) > 0) {
      unknownTextureSize++;
    }
    if (importerMaterial->GetTextureCount(aiTextureType_LIGHTMAP) > 0) {
      unknownTextureSize++;
    }
    if (importerMaterial->GetTextureCount(aiTextureType_REFLECTION) > 0) {
      unknownTextureSize++;
    }
    if (importerMaterial->GetTextureCount(aiTextureType_EMISSION_COLOR) > 0) {
      unknownTextureSize++;
    }
    if (importerMaterial->GetTextureCount(aiTextureType_SHEEN) > 0) {
      unknownTextureSize++;
    }
    if (importerMaterial->GetTextureCount(aiTextureType_CLEARCOAT) > 0) {
      unknownTextureSize++;
    }

    if (importerMaterial->GetTextureCount(aiTextureType_UNKNOWN) > 0) {
      unknownTextureSize++;
    }

    aiColor3D color;
    if (importerMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == aiReturn_SUCCESS) {
      targetMaterial->m_materialProperties.m_albedoColor = glm::vec3(color.r, color.g, color.b);
    } else if (importerMaterial->Get(AI_MATKEY_BASE_COLOR, color) == aiReturn_SUCCESS) {
      targetMaterial->m_materialProperties.m_albedoColor = glm::vec3(color.r, color.g, color.b);
    }
    ai_real factor;
    if (importerMaterial->Get(AI_MATKEY_METALLIC_FACTOR, factor) == aiReturn_SUCCESS) {
      targetMaterial->m_materialProperties.m_metallic = factor;
    }
    if (importerMaterial->Get(AI_MATKEY_ROUGHNESS_FACTOR, factor) == aiReturn_SUCCESS) {
      targetMaterial->m_materialProperties.m_roughness = factor;
    }
    if (importerMaterial->Get(AI_MATKEY_SPECULAR_FACTOR, factor) == aiReturn_SUCCESS) {
      targetMaterial->m_materialProperties.m_specular = factor;
    }
  }
  return targetMaterial;
}
std::shared_ptr<Mesh> ReadMesh(aiMesh* importerMesh) {
  VertexAttributes attributes;
  std::vector<Vertex> vertices;
  std::vector<unsigned> indices;
  if (importerMesh->mNumVertices == 0 || !importerMesh->HasFaces())
    return nullptr;
  vertices.resize(importerMesh->mNumVertices);
  // Walk through each of the mesh's vertices
  for (int i = 0; i < importerMesh->mNumVertices; i++) {
    Vertex vertex;
    glm::vec3 v3;  // we declare a placeholder vector since assimp uses its own vector class that doesn't directly
    // convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
    // positions
    v3.x = importerMesh->mVertices[i].x;
    v3.y = importerMesh->mVertices[i].y;
    v3.z = importerMesh->mVertices[i].z;
    vertex.position = v3;
    if (importerMesh->HasNormals()) {
      v3.x = importerMesh->mNormals[i].x;
      v3.y = importerMesh->mNormals[i].y;
      v3.z = importerMesh->mNormals[i].z;
      vertex.normal = v3;
      attributes.normal = true;
    } else {
      attributes.normal = false;
    }
    if (importerMesh->HasTangentsAndBitangents()) {
      v3.x = importerMesh->mTangents[i].x;
      v3.y = importerMesh->mTangents[i].y;
      v3.z = importerMesh->mTangents[i].z;
      vertex.m_tangent = v3;
      attributes.tangent = true;
    } else {
      attributes.tangent = false;
    }
    if (importerMesh->HasVertexColors(0)) {
      v3.x = importerMesh->mColors[0][i].r;
      v3.y = importerMesh->mColors[0][i].g;
      v3.z = importerMesh->mColors[0][i].b;
      vertex.color = glm::vec4(v3, 1.0f);
      attributes.color = true;
    } else {
      attributes.color = false;
    }
    if (importerMesh->HasTextureCoords(0)) {
      glm::vec2 v2;
      v2.x = importerMesh->mTextureCoords[0][i].x;
      v2.y = importerMesh->mTextureCoords[0][i].y;
      vertex.tex_coord = v2;
      attributes.tex_coord = true;
    } else {
      vertex.tex_coord = glm::vec2(0.0f, 0.0f);
      attributes.color = false;
    }
    vertices[i] = vertex;
  }
  // now walk through each of the mesh's _Faces (a face is a mesh its triangle) and retrieve the corresponding vertex
  // indices.
  for (int i = 0; i < importerMesh->mNumFaces; i++) {
    assert(importerMesh->mFaces[i].mNumIndices == 3);
    // retrieve all indices of the face and store them in the indices vector
    for (int j = 0; j < 3; j++)
      indices.push_back(importerMesh->mFaces[i].mIndices[j]);
  }
  auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
  mesh->SetVertices(attributes, vertices, indices);
  return mesh;
}
std::shared_ptr<SkinnedMesh> ReadSkinnedMesh(std::unordered_map<std::string, std::shared_ptr<Bone>>& bonesMap,
                                             aiMesh* importerMesh) {
  SkinnedVertexAttributes skinnedVertexAttributes{};
  std::vector<SkinnedVertex> vertices;
  std::vector<unsigned> indices;
  if (importerMesh->mNumVertices == 0 || !importerMesh->HasFaces())
    return nullptr;
  vertices.resize(importerMesh->mNumVertices);
  // Walk through each of the mesh's vertices
  for (int i = 0; i < importerMesh->mNumVertices; i++) {
    SkinnedVertex vertex;
    glm::vec3 v3;  // we declare a placeholder vector since assimp uses its own vector class that doesn't directly
    // convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
    // positions
    v3.x = importerMesh->mVertices[i].x;
    v3.y = importerMesh->mVertices[i].y;
    v3.z = importerMesh->mVertices[i].z;
    vertex.position = v3;
    if (importerMesh->HasNormals()) {
      v3.x = importerMesh->mNormals[i].x;
      v3.y = importerMesh->mNormals[i].y;
      v3.z = importerMesh->mNormals[i].z;
      vertex.normal = v3;
      skinnedVertexAttributes.normal = true;
    }
    if (importerMesh->HasTangentsAndBitangents()) {
      v3.x = importerMesh->mTangents[i].x;
      v3.y = importerMesh->mTangents[i].y;
      v3.z = importerMesh->mTangents[i].z;
      vertex.tangent = v3;
      skinnedVertexAttributes.tangent = true;
    }
    if (importerMesh->HasVertexColors(0)) {
      v3.x = importerMesh->mColors[0][i].r;
      v3.y = importerMesh->mColors[0][i].g;
      v3.z = importerMesh->mColors[0][i].b;
      vertex.color = glm::vec4(v3, 1.0f);
      skinnedVertexAttributes.color = true;
    }
    glm::vec2 v2;
    if (importerMesh->HasTextureCoords(0)) {
      v2.x = importerMesh->mTextureCoords[0][i].x;
      v2.y = importerMesh->mTextureCoords[0][i].y;
      vertex.tex_coord = v2;
      skinnedVertexAttributes.tex_coord = true;
    } else {
      vertex.tex_coord = glm::vec2(0.0f, 0.0f);
      skinnedVertexAttributes.tex_coord = true;
    }
    vertices[i] = vertex;
  }
  // now walk through each of the mesh's _Faces (a face is a mesh its triangle) and retrieve the corresponding vertex
  // indices.
  for (int i = 0; i < importerMesh->mNumFaces; i++) {
    assert(importerMesh->mFaces[i].mNumIndices == 3);
    // retrieve all indices of the face and store them in the indices vector
    for (int j = 0; j < 3; j++)
      indices.push_back(importerMesh->mFaces[i].mIndices[j]);
  }
  auto skinnedMesh = ProjectManager::CreateTemporaryAsset<SkinnedMesh>();
#pragma region Read bones
  std::vector<std::vector<std::pair<int, float>>> verticesBoneIdWeights;
  verticesBoneIdWeights.resize(vertices.size());
  for (unsigned i = 0; i < importerMesh->mNumBones; i++) {
    aiBone* importerBone = importerMesh->mBones[i];
    auto name = importerBone->mName.C_Str();
    if (const auto search = bonesMap.find(name); search == bonesMap.end())  // If we can't find this bone
    {
      std::shared_ptr<Bone> bone = std::make_shared<Bone>();
      bone->name = name;
      bone->offset_matrix.value = mat4_cast(importerBone->mOffsetMatrix);
      bonesMap[name] = bone;
      skinnedMesh->bones.push_back(bone);
    } else {
      skinnedMesh->bones.push_back(search->second);
    }

    for (int j = 0; j < importerBone->mNumWeights; j++) {
      verticesBoneIdWeights[importerBone->mWeights[j].mVertexId].emplace_back(i, importerBone->mWeights[j].mWeight);
    }
  }
  for (unsigned i = 0; i < verticesBoneIdWeights.size(); i++) {
    auto ids = glm::ivec4(-1);
    auto weights = glm::vec4(0.0f);
    auto& list = verticesBoneIdWeights[i];
    for (unsigned j = 0; j < 4; j++) {
      if (!list.empty()) {
        int extract = -1;
        float max = -1.0f;
        for (int k = 0; k < list.size(); k++) {
          if (list[k].second > max) {
            max = list[k].second;
            extract = k;
          }
        }
        ids[j] = list[extract].first;
        weights[j] = list[extract].second;
        list.erase(list.begin() + extract);
      } else
        break;
    }
    vertices[i].bond_id = ids;
    vertices[i].weight = weights;

    ids = glm::ivec4(-1);
    weights = glm::vec4(0.0f);
    for (unsigned j = 0; j < 4; j++) {
      if (!list.empty()) {
        int extract = -1;
        float max = -1.0f;
        for (int k = 0; k < list.size(); k++) {
          if (list[k].second > max) {
            max = list[k].second;
            extract = k;
          }
        }
        ids[j] = list[extract].first;
        weights[j] = list[extract].second;
        list.erase(list.begin() + extract);
      } else
        break;
    }
    vertices[i].bond_id2 = ids;
    vertices[i].weight2 = weights;
  }
#pragma endregion
  skinnedMesh->SetVertices(skinnedVertexAttributes, vertices, indices);
  return skinnedMesh;
}

bool ProcessNode(const std::string& directory, Prefab* modelNode,
                 std::unordered_map<unsigned, std::shared_ptr<Material>>& loadedMaterials,
                 std::unordered_map<std::string, std::shared_ptr<Texture2D>>& texture2DsLoaded,
                 std::vector<std::pair<std::shared_ptr<Texture2D>, std::shared_ptr<Texture2D>>>& opacityMaps,
                 std::unordered_map<std::string, std::shared_ptr<Bone>>& bonesMap, const aiNode* importerNode,
                 const std::shared_ptr<AssimpImportNode>& assimpNode, const aiScene* importerScene,
                 const std::shared_ptr<Animation>& animation) {
  bool addedMeshRenderer = false;
  for (unsigned i = 0; i < importerNode->mNumMeshes; i++) {
    // the modelNode object only contains indices to index the actual objects in the scene.
    // the scene contains all the data, modelNode is just to keep stuff organized (like relations between nodes).
    aiMesh* importerMesh = importerScene->mMeshes[importerNode->mMeshes[i]];
    if (!importerMesh)
      continue;
    auto childNode = ProjectManager::CreateTemporaryAsset<Prefab>();
    childNode->instance_name = std::string(importerMesh->mName.C_Str());
    const auto search = loadedMaterials.find(importerMesh->mMaterialIndex);
    bool isSkinnedMesh = importerMesh->mNumBones != 0xffffffff && importerMesh->mBones;
    std::shared_ptr<Material> material;
    if (search == loadedMaterials.end()) {
      aiMaterial* importerMaterial = nullptr;
      if (importerMesh->mMaterialIndex != 0xffffffff && importerMesh->mMaterialIndex < importerScene->mNumMaterials)
        importerMaterial = importerScene->mMaterials[importerMesh->mMaterialIndex];
      material = ReadMaterial(directory, texture2DsLoaded, opacityMaps, importerMaterial);
      loadedMaterials[importerMesh->mMaterialIndex] = material;
    } else {
      material = search->second;
    }

    if (isSkinnedMesh) {
      auto skinnedMeshRenderer = Serialization::ProduceSerializable<SkinnedMeshRenderer>();
      skinnedMeshRenderer->m_material.Set<Material>(material);
      skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(ReadSkinnedMesh(bonesMap, importerMesh));
      if (!skinnedMeshRenderer->m_skinnedMesh.Get())
        continue;
      addedMeshRenderer = true;
      PrivateComponentHolder holder;
      holder.enabled = true;
      holder.private_component = std::static_pointer_cast<IPrivateComponent>(skinnedMeshRenderer);
      childNode->private_components.push_back(holder);
    } else {
      auto meshRenderer = Serialization::ProduceSerializable<MeshRenderer>();
      meshRenderer->m_material.Set<Material>(material);
      meshRenderer->m_mesh.Set<Mesh>(ReadMesh(importerMesh));
      if (!meshRenderer->m_mesh.Get())
        continue;
      addedMeshRenderer = true;
      PrivateComponentHolder holder;
      holder.enabled = true;
      holder.private_component = std::static_pointer_cast<IPrivateComponent>(meshRenderer);
      childNode->private_components.push_back(holder);
    }
    auto transform = std::make_shared<Transform>();
    transform->value = mat4_cast(importerNode->mTransformation);
    if (!importerNode->mParent)
      transform->value = Transform().value;

    DataComponentHolder holder;
    holder.data_component_type = Typeof<Transform>();
    holder.data_component = transform;
    childNode->data_components.push_back(holder);

    modelNode->child_prefabs.push_back(std::move(childNode));
  }

  for (unsigned i = 0; i < importerNode->mNumChildren; i++) {
    auto childNode = ProjectManager::CreateTemporaryAsset<Prefab>();
    childNode->instance_name = std::string(importerNode->mChildren[i]->mName.C_Str());
    auto childAssimpNode = std::make_shared<AssimpImportNode>(importerNode->mChildren[i]);
    childAssimpNode->m_parent = assimpNode;
    const bool childAdd = ProcessNode(directory, childNode.get(), loadedMaterials, texture2DsLoaded, opacityMaps,
                                      bonesMap, importerNode->mChildren[i], childAssimpNode, importerScene, animation);
    if (childAdd) {
      modelNode->child_prefabs.push_back(std::move(childNode));
    }
    addedMeshRenderer = addedMeshRenderer | childAdd;
    assimpNode->m_children.push_back(std::move(childAssimpNode));
  }
  return addedMeshRenderer;
}
#pragma endregion

#pragma region Model Loading
void Prefab::AttachChildrenPrivateComponent(const std::shared_ptr<Scene>& scene,
                                            const std::shared_ptr<Prefab>& model_node, const Entity& parent_entity,
                                            const std::unordered_map<Handle, Handle>& map) const {
  Entity entity;
  auto children = scene->GetChildren(parent_entity);
  for (auto& i : children) {
    auto a = scene->GetEntityHandle(i).GetValue();
    auto b = map.at(model_node->entity_handle).GetValue();
    if (a == b)
      entity = i;
  }
  if (entity.GetIndex() == 0)
    return;
  for (auto& i : model_node->private_components) {
    size_t id;
    auto ptr = std::static_pointer_cast<IPrivateComponent>(
        Serialization::ProduceSerializable(i.private_component->GetTypeName(), id));
    Serialization::ClonePrivateComponent(ptr, i.private_component);
    ptr->scene_ = scene;
    scene->SetPrivateComponent(entity, ptr);
  }
  int index = 0;
  for (auto& i : model_node->child_prefabs) {
    AttachChildrenPrivateComponent(scene, i, entity, map);
    index++;
  }
  scene->SetEnable(entity, enabled_);
}
void Prefab::AttachChildren(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Prefab>& model_node,
                            Entity parent_entity, std::unordered_map<Handle, Handle>& map) {
  std::vector<DataComponentType> types;
  for (auto& i : model_node->data_components) {
    types.emplace_back(i.data_component_type);
  }
  auto archetype = Entities::CreateEntityArchetype("", types);
  auto entity = scene->CreateEntity(archetype, model_node->instance_name);
  map[model_node->entity_handle] = scene->GetEntityHandle(entity);
  scene->SetParent(entity, parent_entity);
  for (auto& i : model_node->data_components) {
    scene->SetDataComponent(entity.GetIndex(), i.data_component_type.type_index, i.data_component_type.type_size,
                            i.data_component.get());
  }
  int index = 0;
  for (auto& i : model_node->child_prefabs) {
    AttachChildren(scene, i, entity, map);
    index++;
  }
}

void Prefab::AttachAnimator(Prefab* parent, const Handle& animator_entity_handle) {
  if (const auto skinnedMeshRenderer = parent->GetPrivateComponent<SkinnedMeshRenderer>()) {
    skinnedMeshRenderer->m_animator.entity_handle_ = animator_entity_handle;
    skinnedMeshRenderer->m_animator.private_component_type_name_ = "Animator";
  }
  for (auto& i : parent->child_prefabs) {
    AttachAnimator(i.get(), animator_entity_handle);
  }
}
void Prefab::ApplyBoneIndices(Prefab* node) {
  if (const auto skinnedMeshRenderer = node->GetPrivateComponent<SkinnedMeshRenderer>()) {
    skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>()->FetchIndices();
    skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>()->bones.clear();
  }
  for (auto& i : node->child_prefabs) {
    ApplyBoneIndices(i.get());
  }
}
void Prefab::FromEntity(const Entity& entity) {
  const auto scene = Application::GetActiveScene();
  if (!scene) {
    EVOENGINE_ERROR("Scene not attached!");
    return;
  }
  entity_handle = scene->GetEntityHandle(entity);
  instance_name = scene->GetEntityName(entity);
  enabled_ = scene->IsEntityEnabled(entity);
  scene->UnsafeForEachDataComponent(entity, [&](const DataComponentType& type, const void* data) {
    DataComponentHolder holder;
    holder.data_component_type = type;
    size_t id;
    size_t size;
    holder.data_component =
        std::static_pointer_cast<IDataComponent>(Serialization::ProduceDataComponent(type.type_name, id, size));
    memcpy(holder.data_component.get(), data, type.type_size);
    data_components.push_back(std::move(holder));
  });

  const auto& elements =
      scene->scene_data_storage_.entity_metadata_list.at(entity.GetIndex()).private_component_elements;
  for (auto& element : elements) {
    size_t id;
    auto ptr = std::static_pointer_cast<IPrivateComponent>(
        Serialization::ProduceSerializable(element.private_component_data->GetTypeName(), id));
    ptr->OnCreate();
    Serialization::ClonePrivateComponent(ptr, element.private_component_data);
    PrivateComponentHolder holder;
    holder.enabled = element.private_component_data->enabled_;
    holder.private_component = ptr;
    private_components.push_back(holder);
  }

  const auto children = scene->GetChildren(entity);
  for (auto& i : children) {
    auto temp = ProjectManager::CreateTemporaryAsset<Prefab>();
    temp->instance_name = scene->GetEntityName(i);
    child_prefabs.push_back(temp);
    child_prefabs.back()->FromEntity(i);
  }
}
bool Prefab::LoadInternal(const std::filesystem::path& path) {
  if (path.extension() == ".eveprefab") {
    std::ifstream stream(path.string());
    std::stringstream stringStream;
    stringStream << stream.rdbuf();
    YAML::Node in = YAML::Load(stringStream.str());
#pragma region Assets
    if (const auto& inLocalAssets = in["LocalAssets"]) {
      std::vector<std::shared_ptr<IAsset>> localAssets;
      for (const auto& i : inLocalAssets) {
        Handle handle = i["Handle"].as<uint64_t>();
        localAssets.push_back(ProjectManager::CreateTemporaryAsset(i["TypeName"].as<std::string>(), handle));
      }
      int index = 0;
      for (const auto& i : inLocalAssets) {
        localAssets[index++]->Deserialize(i);
      }
    }

#pragma endregion
    Deserialize(in);
    return true;
  }
  return LoadModelInternal(path);
}
bool Prefab::LoadModelInternal(const std::filesystem::path& path, bool optimize, unsigned int flags) {
  flags = flags | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate;
  if (optimize) {
    flags = flags | aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes;
  }
  // read file via ASSIMP
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(path.string(), flags);
  // check for errors
  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)  // if is Not Zero
  {
    EVOENGINE_LOG("Assimp: " + std::string(importer.GetErrorString()));
    return false;
  }
  // retrieve the directory path of the filepath
  auto temp = path;
  const std::string directory = temp.remove_filename().string();
  std::unordered_map<std::string, std::shared_ptr<Texture2D>> loadedTextures;
  instance_name = path.filename().string();
  std::unordered_map<unsigned, std::shared_ptr<Material>> loadedMaterials;
  std::vector<std::pair<std::shared_ptr<Texture2D>, std::shared_ptr<Texture2D>>> opacityMaps;
  std::unordered_map<std::string, std::shared_ptr<Bone>> bonesMap;
  std::shared_ptr<Animation> animation;
  if (!bonesMap.empty() || scene->HasAnimations()) {
    animation = ProjectManager::CreateTemporaryAsset<Animation>();
  }
  std::shared_ptr<AssimpImportNode> rootAssimpNode = std::make_shared<AssimpImportNode>(scene->mRootNode);
  if (!ProcessNode(directory, this, loadedMaterials, loadedTextures, opacityMaps, bonesMap, scene->mRootNode,
                   rootAssimpNode, scene, animation)) {
    EVOENGINE_ERROR("Model is empty!");
    return false;
  }

  for (auto& pair : opacityMaps) {
    std::vector<glm::vec4> colorData;
    const auto& albedoTexture = pair.first;
    const auto& opacityTexture = pair.second;
    if (!albedoTexture || !opacityTexture)
      continue;
    albedoTexture->GetRgbaChannelData(colorData);
    std::vector<glm::vec4> alphaData;
    const auto resolution = albedoTexture->GetResolution();
    opacityTexture->GetRgbaChannelData(alphaData, resolution.x, resolution.y);
    Jobs::RunParallelFor(colorData.size(), [&](unsigned i) {
      colorData[i].a = alphaData[i].r;
    });
    std::shared_ptr<Texture2D> replacementTexture = ProjectManager::CreateTemporaryAsset<Texture2D>();
    replacementTexture->SetRgbaChannelData(colorData, albedoTexture->GetResolution());
    pair.second = replacementTexture;
  }

  for (const auto& material : loadedMaterials) {
    const auto albedoTexture = material.second->GetAlbedoTexture();
    if (!albedoTexture)
      continue;
    for (const auto& pair : opacityMaps) {
      if (albedoTexture->GetHandle() == pair.first->GetHandle()) {
        material.second->SetAlbedoTexture(pair.second);
      }
    }
  }

  if (!bonesMap.empty() || scene->HasAnimations()) {
    rootAssimpNode->NecessaryWalker(bonesMap);
    size_t index = 0;
    rootAssimpNode->AttachToAnimator(animation, index);
    animation->bone_size = index + 1;
    ReadAnimations(scene, animation, bonesMap);
    ApplyBoneIndices(this);

    auto animator = Serialization::ProduceSerializable<Animator>();
    animator->Setup(animation);
    AttachAnimator(this, entity_handle);
    PrivateComponentHolder holder;
    holder.enabled = true;
    holder.private_component = std::static_pointer_cast<IPrivateComponent>(animator);
    private_components.push_back(holder);
  }
  GatherAssets();
}

#pragma endregion

#pragma region Assimp Export

struct AssimpExportNode {
  int m_meshIndex = -1;
  aiMatrix4x4 m_transform;

  std::vector<AssimpExportNode> m_children;

  void Collect(const std::shared_ptr<Prefab>& currentPrefab, std::vector<std::pair<std::shared_ptr<Mesh>, int>>& meshes,
               std::vector<std::shared_ptr<Material>>& materials);

  void Process(aiNode* exporterNode);
};

void AssimpExportNode::Collect(const std::shared_ptr<Prefab>& currentPrefab,
                               std::vector<std::pair<std::shared_ptr<Mesh>, int>>& meshes,
                               std::vector<std::shared_ptr<Material>>& materials) {
  m_meshIndex = -1;
  for (const auto& dataComponent : currentPrefab->data_components) {
    if (dataComponent.data_component_type == Typeof<Transform>()) {
      m_transform = mat4_cast(std::reinterpret_pointer_cast<Transform>(dataComponent.data_component)->value);
    }
  }
  for (const auto& privateComponent : currentPrefab->private_components) {
    if (const auto meshRenderer = std::dynamic_pointer_cast<MeshRenderer>(privateComponent.private_component)) {
      auto mesh = meshRenderer->m_mesh.Get<Mesh>();
      auto material = meshRenderer->m_material.Get<Material>();
      if (mesh && material) {
        int targetMaterialIndex = -1;
        for (int materialIndex = 0; materialIndex < materials.size(); materialIndex++) {
          if (materials[materialIndex] == material) {
            targetMaterialIndex = materialIndex;
          }
        }
        if (targetMaterialIndex == -1) {
          targetMaterialIndex = materials.size();
          materials.emplace_back(material);
        }

        if (m_meshIndex == -1) {
          m_meshIndex = meshes.size();
          meshes.emplace_back(std::make_pair(mesh, targetMaterialIndex));
        }
      }
    }
  }

  for (const auto& childPrefab : currentPrefab->child_prefabs) {
    m_children.emplace_back();
    auto& newNode = m_children.back();
    newNode.Collect(childPrefab, meshes, materials);
  }
}

void AssimpExportNode::Process(aiNode* exporterNode) {
  if (m_meshIndex != -1) {
    exporterNode->mNumMeshes = 1;
    exporterNode->mMeshes = new unsigned int[1];
    exporterNode->mMeshes[0] = m_meshIndex;
  }
  exporterNode->mNumChildren = m_children.size();
  if (m_children.empty()) {
    exporterNode->mChildren = nullptr;
  } else {
    exporterNode->mChildren = new aiNode*[m_children.size()];
  }
  for (int i = 0; i < m_children.size(); i++) {
    exporterNode->mChildren[i] = new aiNode();
    exporterNode->mChildren[i]->mParent = exporterNode;
    m_children.at(i).Process(exporterNode->mChildren[i]);
  }
}

bool Prefab::SaveModelInternal(const std::filesystem::path& path) const {
  Assimp::Exporter exporter;
  aiScene exporterScene{};
  exporterScene.mMetaData = new aiMetadata();
  std::vector<std::pair<std::shared_ptr<Mesh>, int>> meshes;
  std::vector<std::shared_ptr<Material>> materials;

  AssimpExportNode rootNode;
  rootNode.Collect(std::dynamic_pointer_cast<Prefab>(GetSelf()), meshes, materials);

  exporterScene.mRootNode = new aiNode();
  exporterScene.mNumMeshes = meshes.size();
  if (meshes.empty()) {
    exporterScene.mMeshes = nullptr;
  } else {
    exporterScene.mMeshes = new aiMesh*[meshes.size()];
  }
  for (int meshIndex = 0; meshIndex < meshes.size(); meshIndex++) {
    aiMesh* exporterMesh = exporterScene.mMeshes[meshIndex] = new aiMesh();
    auto& mesh = meshes.at(meshIndex);
    const auto& vertices = mesh.first->UnsafeGetVertices();
    const auto& triangles = mesh.first->UnsafeGetTriangles();
    exporterMesh->mNumVertices = vertices.size();
    exporterMesh->mVertices = new aiVector3D[vertices.size()];
    exporterMesh->mNormals = new aiVector3D[vertices.size()];
    exporterMesh->mNumUVComponents[0] = 2;
    exporterMesh->mTextureCoords[0] = new aiVector3D[vertices.size()];
    exporterMesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
    for (int vertexIndex = 0; vertexIndex < vertices.size(); vertexIndex++) {
      exporterMesh->mVertices[vertexIndex].x = vertices.at(vertexIndex).position.x;
      exporterMesh->mVertices[vertexIndex].y = vertices.at(vertexIndex).position.y;
      exporterMesh->mVertices[vertexIndex].z = vertices.at(vertexIndex).position.z;

      exporterMesh->mNormals[vertexIndex].x = vertices.at(vertexIndex).normal.x;
      exporterMesh->mNormals[vertexIndex].y = vertices.at(vertexIndex).normal.y;
      exporterMesh->mNormals[vertexIndex].z = vertices.at(vertexIndex).normal.z;

      exporterMesh->mTextureCoords[0][vertexIndex].x = vertices.at(vertexIndex).tex_coord.x;
      exporterMesh->mTextureCoords[0][vertexIndex].y = vertices.at(vertexIndex).tex_coord.y;
      exporterMesh->mTextureCoords[0][vertexIndex].z = 0.f;
    }

    exporterMesh->mNumFaces = triangles.size();
    if (triangles.empty()) {
      exporterMesh->mFaces = nullptr;
    } else {
      exporterMesh->mFaces = new aiFace[triangles.size()];
    }
    for (int triangleIndex = 0; triangleIndex < triangles.size(); triangleIndex++) {
      exporterMesh->mFaces[triangleIndex].mIndices = new unsigned int[3];
      exporterMesh->mFaces[triangleIndex].mNumIndices = 3;
      exporterMesh->mFaces[triangleIndex].mIndices[0] = triangles[triangleIndex][0];
      exporterMesh->mFaces[triangleIndex].mIndices[1] = triangles[triangleIndex][1];
      exporterMesh->mFaces[triangleIndex].mIndices[2] = triangles[triangleIndex][2];
    }
    exporterMesh->mMaterialIndex = mesh.second;
    exporterMesh->mName = std::string("mesh_") + std::to_string(meshIndex);
  }

  exporterScene.mNumMaterials = materials.size();
  if (materials.empty()) {
    exporterScene.mMaterials = nullptr;
  } else {
    exporterScene.mMaterials = new aiMaterial*[materials.size()];
  }

  const auto textureFolderPath = std::filesystem::absolute(path.parent_path() / "textures");
  std::filesystem::create_directories(textureFolderPath);

  struct SeparatedTexturePath {
    aiString m_color;
    bool m_hasOpacity = false;
    aiString m_opacity;
  };

  std::unordered_map<std::shared_ptr<Texture2D>, SeparatedTexturePath> collectedTexture;

  for (int materialIndex = 0; materialIndex < materials.size(); materialIndex++) {
    aiMaterial* exporterMaterial = exporterScene.mMaterials[materialIndex] = new aiMaterial();
    auto& material = materials.at(materialIndex);
    exporterMaterial->mNumProperties = 0;
    auto materialName = aiString(std::string("material_") + std::to_string(materialIndex));

    exporterMaterial->AddProperty(&materialName, AI_MATKEY_NAME);

    if (const auto albedoTexture = material->GetAlbedoTexture()) {
      const auto search = collectedTexture.find(albedoTexture);
      SeparatedTexturePath info{};
      if (search != collectedTexture.end()) {
        info = search->second;
      } else {
        if (albedoTexture->IsTemporary()) {
          std::string diffuseTitle = std::to_string(materialIndex) + "_diffuse.png";
          info.m_color = aiString((std::filesystem::path("textures") / diffuseTitle).string());
          const auto succeed = albedoTexture->Export(textureFolderPath / diffuseTitle);
        } else {
          info.m_color =
              aiString((std::filesystem::path("textures") / albedoTexture->GetAbsolutePath().filename()).string());
          std::filesystem::copy(albedoTexture->GetAbsolutePath(),
                                textureFolderPath / albedoTexture->GetAbsolutePath().filename(),
                                std::filesystem::copy_options::overwrite_existing);
        }
        if (albedoTexture->m_alphaChannel) {
          info.m_hasOpacity = true;
          std::string opacityTitle = std::to_string(materialIndex) + "_opacity.png";
          info.m_opacity = aiString((std::filesystem::path("textures") / opacityTitle).string());
          std::vector<glm::vec4> data;
          albedoTexture->GetRgbaChannelData(data);
          std::vector<float> src(data.size() * 4);
          Jobs::RunParallelFor(data.size(), [&](const unsigned i) {
            src[i * 4] = data[i].a;
            src[i * 4 + 1] = data[i].a;
            src[i * 4 + 2] = data[i].a;
            src[i * 4 + 3] = data[i].a;
          });
          auto resolution = albedoTexture->GetResolution();
          Texture2D::StoreToPng(textureFolderPath / opacityTitle, src, resolution.x, resolution.y, 4, 4);
        }
      }

      exporterMaterial->AddProperty(&info.m_color, AI_MATKEY_TEXTURE_DIFFUSE(0));
      if (info.m_hasOpacity) {
        exporterMaterial->AddProperty(&info.m_opacity, AI_MATKEY_TEXTURE_OPACITY(0));
      }
    }

    if (const auto normalTexture = material->GetNormalTexture()) {
      const auto search = collectedTexture.find(normalTexture);
      SeparatedTexturePath info{};
      if (search != collectedTexture.end()) {
        info = search->second;
      } else {
        if (normalTexture->IsTemporary()) {
          std::string title = std::to_string(materialIndex) + "_normal.png";
          info.m_color = aiString((std::filesystem::path("textures") / title).string());
          const auto succeed = normalTexture->Export(textureFolderPath / title);
        } else {
          info.m_color =
              aiString((std::filesystem::path("textures") / normalTexture->GetAbsolutePath().filename()).string());
          std::filesystem::copy(normalTexture->GetAbsolutePath(),
                                textureFolderPath / normalTexture->GetAbsolutePath().filename(),
                                std::filesystem::copy_options::overwrite_existing);
        }
      }

      exporterMaterial->AddProperty(&info.m_color, AI_MATKEY_TEXTURE_NORMALS(0));
    }
    if (const auto metallicTexture = material->GetMetallicTexture()) {
      const auto search = collectedTexture.find(metallicTexture);
      SeparatedTexturePath info{};
      if (search != collectedTexture.end()) {
        info = search->second;
      } else {
        if (metallicTexture->IsTemporary()) {
          std::string title = std::to_string(materialIndex) + "_metallic.png";
          info.m_color = aiString((std::filesystem::path("textures") / title).string());
          const auto succeed = metallicTexture->Export(textureFolderPath / title);
        } else {
          info.m_color =
              aiString((std::filesystem::path("textures") / metallicTexture->GetAbsolutePath().filename()).string());
          std::filesystem::copy(metallicTexture->GetAbsolutePath(),
                                textureFolderPath / metallicTexture->GetAbsolutePath().filename(),
                                std::filesystem::copy_options::overwrite_existing);
        }
      }
      exporterMaterial->AddProperty(&info.m_color, AI_MATKEY_TEXTURE_SHININESS(0));
    }
    if (const auto roughnessTexture = material->GetRoughnessTexture()) {
      const auto search = collectedTexture.find(roughnessTexture);
      SeparatedTexturePath info{};
      if (search != collectedTexture.end()) {
        info = search->second;
      } else {
        if (roughnessTexture->IsTemporary()) {
          std::string title = std::to_string(materialIndex) + "_roughness.png";
          info.m_color = aiString((std::filesystem::path("textures") / title).string());
          const auto succeed = roughnessTexture->Export(textureFolderPath / title);
        } else {
          info.m_color =
              aiString((std::filesystem::path("textures") / roughnessTexture->GetAbsolutePath().filename()).string());
          std::filesystem::copy(roughnessTexture->GetAbsolutePath(),
                                textureFolderPath / roughnessTexture->GetAbsolutePath().filename(),
                                std::filesystem::copy_options::overwrite_existing);
        }
      }
      exporterMaterial->AddProperty(&info.m_color, AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE_ROUGHNESS, 0));
    }
    if (const auto aoTexture = material->GetAoTexture()) {
      const auto search = collectedTexture.find(aoTexture);
      SeparatedTexturePath info{};
      if (search != collectedTexture.end()) {
        info = search->second;
      } else {
        if (aoTexture->IsTemporary()) {
          std::string title = std::to_string(materialIndex) + "_ao.png";
          info.m_color = aiString((std::filesystem::path("textures") / title).string());
          const auto succeed = aoTexture->Export(textureFolderPath / title);
        } else {
          info.m_color =
              aiString((std::filesystem::path("textures") / aoTexture->GetAbsolutePath().filename()).string());
          std::filesystem::copy(aoTexture->GetAbsolutePath(),
                                textureFolderPath / aoTexture->GetAbsolutePath().filename(),
                                std::filesystem::copy_options::overwrite_existing);
        }
      }
      exporterMaterial->AddProperty(&info.m_color, AI_MATKEY_TEXTURE(aiTextureType_AMBIENT_OCCLUSION, 0));
    }
  }

  std::string formatId;
  if (path.extension().string() == ".obj") {
    formatId = "obj";
  } else if (path.extension().string() == ".fbx") {
    formatId = "fbx";
  } else if (path.extension().string() == ".gltf") {
    formatId = "gltf";
  } else if (path.extension().string() == ".dae") {
    formatId = "dae";
  }

  rootNode.Process(exporterScene.mRootNode);
  exporter.Export(&exporterScene, formatId.c_str(), path.string());

  return true;
}

#pragma endregion

Entity Prefab::ToEntity(const std::shared_ptr<Scene>& scene, bool auto_adjust_size) const {
  std::unordered_map<Handle, Handle> entityMap;
  std::vector<DataComponentType> types;
  types.reserve(data_components.size());
  for (auto& i : data_components) {
    types.emplace_back(i.data_component_type);
  }
  auto archetype = Entities::CreateEntityArchetype("", types);
  const Entity entity = scene->CreateEntity(archetype, instance_name);
  entityMap[entity_handle] = scene->GetEntityHandle(entity);
  for (auto& i : data_components) {
    scene->SetDataComponent(entity.GetIndex(), i.data_component_type.type_index, i.data_component_type.type_size,
                            i.data_component.get());
  }
  int index = 0;
  for (const auto& i : child_prefabs) {
    AttachChildren(scene, i, entity, entityMap);
    index++;
  }

  for (auto& i : private_components) {
    size_t id;
    auto ptr = std::static_pointer_cast<IPrivateComponent>(
        Serialization::ProduceSerializable(i.private_component->GetTypeName(), id));
    Serialization::ClonePrivateComponent(ptr, i.private_component);
    ptr->handle_ = Handle();
    ptr->scene_ = scene;
    scene->SetPrivateComponent(entity, ptr);
  }
  for (const auto& i : child_prefabs) {
    AttachChildrenPrivateComponent(scene, i, entity, entityMap);
    index++;
  }

  RelinkChildren(scene, entity, entityMap);

  scene->SetEnable(entity, enabled_);

  TransformGraph::CalculateTransformGraphForDescendants(scene, entity);

  if (auto_adjust_size) {
    const auto bound = scene->GetEntityBoundingBox(entity);
    auto size = bound.Size();
    glm::vec3 scale = glm::vec3(1.f);
    while (size.x > 10.f || size.y > 10.f || size.z > 10.f) {
      scale /= 2.f;
      size /= 2.f;
    }
    Transform t{};
    GlobalTransform gt{};
    gt.SetScale(scale);
    t.SetScale(scale);
    scene->SetDataComponent(entity, t);
    scene->SetDataComponent(entity, gt);
    TransformGraph::CalculateTransformGraphForDescendants(scene, entity);
  }
  return entity;
}

void DataComponentHolder::Serialize(YAML::Emitter& out) const {
  out << YAML::Key << "m_type.type_name" << YAML::Value << data_component_type.type_name;
  out << YAML::Key << "data_component" << YAML::Value
      << YAML::Binary((const unsigned char*)data_component.get(), data_component_type.type_size);
}
bool DataComponentHolder::Deserialize(const YAML::Node& in) {
  data_component_type.type_name = in["m_type.type_name"].as<std::string>();
  if (!Serialization::HasComponentDataType(data_component_type.type_name))
    return false;
  data_component = Serialization::ProduceDataComponent(data_component_type.type_name, data_component_type.type_index,
                                                       data_component_type.type_size);
  if (in["data_component"]) {
    YAML::Binary data = in["data_component"].as<YAML::Binary>();
    std::memcpy(data_component.get(), data.data(), data.size());
  }
  return true;
}
void Prefab::Serialize(YAML::Emitter& out) const {
  out << YAML::Key << "instance_name" << YAML::Value << instance_name;
  out << YAML::Key << "enabled_" << YAML::Value << enabled_;
  out << YAML::Key << "entity_handle_" << YAML::Value << entity_handle.GetValue();

  if (!data_components.empty()) {
    out << YAML::Key << "data_components" << YAML::BeginSeq;
    for (auto& i : data_components) {
      out << YAML::BeginMap;
      i.Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }

  if (!private_components.empty()) {
    out << YAML::Key << "private_components" << YAML::BeginSeq;
    for (auto& i : private_components) {
      out << YAML::BeginMap;
      i.Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }

  if (!child_prefabs.empty()) {
    out << YAML::Key << "child_prefabs" << YAML::BeginSeq;
    for (auto& i : child_prefabs) {
      out << YAML::BeginMap;
      out << YAML::Key << "m_handle" << i->GetHandle().GetValue();
      i->Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }
}
void Prefab::Deserialize(const YAML::Node& in) {
  instance_name = in["instance_name"].as<std::string>();
  enabled_ = in["enabled_"].as<bool>();
  entity_handle = Handle(in["entity_handle_"].as<uint64_t>());
  if (in["data_components"]) {
    for (const auto& i : in["data_components"]) {
      DataComponentHolder holder;
      if (holder.Deserialize(i)) {
        data_components.push_back(holder);
      }
    }
  }

  std::vector<std::pair<int, std::shared_ptr<IAsset>>> localAssets;
  if (const auto inLocalAssets = in["LocalAssets"]) {
    int index = 0;
    for (const auto& i : inLocalAssets) {
      // First, find the asset in assetregistry
      if (const auto typeName = i["TypeName"].as<std::string>(); Serialization::HasSerializableType(typeName)) {
        auto asset = ProjectManager::CreateTemporaryAsset(typeName, i["Handle"].as<uint64_t>());
        localAssets.emplace_back(index, asset);
      }
      index++;
    }

    for (const auto& i : localAssets) {
      i.second->Deserialize(inLocalAssets[i.first]);
    }
  }

  if (in["private_components"]) {
    for (const auto& i : in["private_components"]) {
      PrivateComponentHolder holder;
      holder.Deserialize(i);
      private_components.push_back(holder);
    }
  }

  if (in["child_prefabs"]) {
    for (const auto& i : in["child_prefabs"]) {
      auto child = ProjectManager::CreateTemporaryAsset<Prefab>();
      child->handle_ = i["m_handle"].as<uint64_t>();
      child->Deserialize(i);
      child_prefabs.push_back(child);
    }
  }
}
void Prefab::CollectAssets(std::unordered_map<Handle, std::shared_ptr<IAsset>>& map) const {
  std::vector<AssetRef> list;
  for (auto& i : private_components) {
    i.private_component->CollectAssetRef(list);
  }
  for (auto& i : list) {
    auto asset = i.Get<IAsset>();
    if (asset && !Resources::IsResource(asset) && asset->IsTemporary()) {
      map[asset->GetHandle()] = asset;
    }
  }
  bool listCheck = true;
  while (listCheck) {
    size_t currentSize = map.size();
    list.clear();
    for (auto& i : map) {
      i.second->CollectAssetRef(list);
    }
    for (auto& i : list) {
      auto asset = i.Get<IAsset>();
      if (asset && !Resources::IsResource(asset) && asset->IsTemporary()) {
        map[asset->GetHandle()] = asset;
      }
    }
    if (map.size() == currentSize)
      listCheck = false;
  }
  for (auto& i : child_prefabs)
    i->CollectAssets(map);
}
bool Prefab::SaveInternal(const std::filesystem::path& path) const {
  if (path.extension() == ".eveprefab") {
    auto directory = path;
    directory.remove_filename();
    std::filesystem::create_directories(directory);
    YAML::Emitter out;
    out << YAML::BeginMap;
    Serialize(out);
    std::unordered_map<Handle, std::shared_ptr<IAsset>> assetMap;
    CollectAssets(assetMap);
    if (!assetMap.empty()) {
      out << YAML::Key << "LocalAssets" << YAML::Value << YAML::BeginSeq;
      for (auto& i : assetMap) {
        if (!i.second->Saved())
          i.second->Save();
        out << YAML::BeginMap;
        out << YAML::Key << "TypeName" << YAML::Value << i.second->GetTypeName();
        out << YAML::Key << "Handle" << YAML::Value << i.first.GetValue();
        i.second->Serialize(out);
        out << YAML::EndMap;
      }
      out << YAML::EndSeq;
    }
    out << YAML::EndMap;

    std::ofstream fout(path.string());
    fout << out.c_str();
    fout.flush();
    return true;
  }

  return SaveModelInternal(path);
}
void Prefab::RelinkChildren(const std::shared_ptr<Scene>& scene, const Entity& parent_entity,
                            const std::unordered_map<Handle, Handle>& map) {
  scene->ForEachPrivateComponent(parent_entity, [&](PrivateComponentElement& data) {
    data.private_component_data->Relink(map, scene);
  });
  scene->ForEachChild(parent_entity, [&](Entity child) {
    RelinkChildren(scene, child, map);
  });
}

void Prefab::LoadModel(const std::filesystem::path& path, const bool optimize, const unsigned flags) {
  LoadModelInternal(ProjectManager::GetProjectPath().parent_path() / path, optimize, flags);
}

void Prefab::GatherAssets() {
  collected_assets.clear();
  for (const auto& components : private_components) {
    std::vector<AssetRef> assetRefs;
    components.private_component->CollectAssetRef(assetRefs);
    for (const auto& assetRef : assetRefs)
      collected_assets[assetRef.GetAssetHandle()] = assetRef;
  }

  for (const auto& child : child_prefabs)
    GatherAssetsWalker(child, collected_assets);
}

bool Prefab::OnInspectWalker(const std::shared_ptr<Prefab>& walker) {
  bool changed = false;
  ImGui::Text(("Name: " + walker->instance_name).c_str());

  if (OnInspectComponents(walker))
    changed = true;

  if (!walker->child_prefabs.empty()) {
    if (ImGui::TreeNode("Children")) {
      for (const auto& child : walker->child_prefabs) {
        if (OnInspectWalker(child))
          changed = true;
      }
      ImGui::TreePop();
    }
  }
  return changed;
}

void Prefab::GatherAssetsWalker(const std::shared_ptr<Prefab>& walker, std::unordered_map<Handle, AssetRef>& assets) {
  for (const auto& i : walker->private_components) {
    std::vector<AssetRef> assetRefs;
    i.private_component->CollectAssetRef(assetRefs);
    for (const auto& assetRef : assetRefs)
      assets[assetRef.GetAssetHandle()] = assetRef;
  }

  for (const auto& child : walker->child_prefabs)
    GatherAssetsWalker(child, assets);
}

bool Prefab::OnInspectComponents(const std::shared_ptr<Prefab>& walker) {
  bool changed = false;
  if (ImGui::TreeNode("Data Components")) {
    for (auto& i : walker->data_components) {
      ImGui::Text(("Type: " + i.data_component_type.type_name).c_str());
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Private Components")) {
    for (auto& i : walker->private_components) {
      ImGui::Text(("Type: " + i.private_component->GetTypeName()).c_str());
    }
    ImGui::TreePop();
  }
  return changed;
}
bool Prefab::OnInspect(const std::shared_ptr<EditorLayer>& editor_layer) {
  bool changed = false;
  if (ImGui::Button("Instantiate")) {
    ToEntity(Application::GetActiveScene());
  }
  if (collected_assets.empty() && ImGui::Button("Collect assets"))
    GatherAssets();
  if (!collected_assets.empty()) {
    if (ImGui::TreeNode("Assets")) {
      for (auto& i : collected_assets) {
        const auto ptr = i.second.Get<IAsset>();
        const std::string tag = "##" + ptr->GetTypeName() + std::to_string(ptr->GetHandle());
        ImGui::Button((ptr->GetTitle() + tag).c_str());
        EditorLayer::DraggableAsset(ptr);
        EditorLayer::Rename(i.second);
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
          ProjectManager::GetInstance().inspecting_asset = ptr;
        }
      }
      ImGui::TreePop();
    }
  }

  if (ImGui::TreeNode("Prefab Hierarchy")) {
    ImGui::Text((instance_name + " (root)").c_str());
    if (OnInspectComponents(std::dynamic_pointer_cast<Prefab>(GetSelf())))
      changed = true;

    if (!child_prefabs.empty()) {
      if (ImGui::TreeNode("Children")) {
        for (const auto& child : child_prefabs) {
          if (OnInspectWalker(child))
            changed = true;
        }
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }
  return changed;
}

void PrivateComponentHolder::Serialize(YAML::Emitter& out) const {
  out << YAML::Key << "enabled" << YAML::Value << enabled;
  out << YAML::Key << "type_name" << YAML::Value << private_component->GetTypeName();

  out << YAML::Key << "private_component" << YAML::BeginMap;
  out << YAML::Key << "handle" << private_component->GetHandle().GetValue();
  private_component->Serialize(out);
  out << YAML::EndMap;
}
void PrivateComponentHolder::Deserialize(const YAML::Node& in) {
  enabled = in["enabled"].as<bool>();
  auto typeName = in["type_name"].as<std::string>();
  auto inData = in["private_component"];
  if (Serialization::HasSerializableType(typeName)) {
    size_t hashCode;
    private_component = std::dynamic_pointer_cast<IPrivateComponent>(
        Serialization::ProduceSerializable(typeName, hashCode, Handle(inData["handle"].as<uint64_t>())));
  } else {
    size_t hashCode;
    private_component = std::dynamic_pointer_cast<IPrivateComponent>(Serialization::ProduceSerializable(
        "UnknownPrivateComponent", hashCode, Handle(inData["handle"].as<uint64_t>())));
    std::dynamic_pointer_cast<UnknownPrivateComponent>(private_component)->m_originalTypeName = typeName;
  }
  private_component->OnCreate();
  private_component->Deserialize(inData);
}
