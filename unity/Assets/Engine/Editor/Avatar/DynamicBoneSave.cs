using System.Collections.Generic;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace XEditor
{
    public class DynamicBoneSave : MonoBehaviour
    {
        public static string Save(GameObject go, GameObject save)
        {
            DynamicBoneSave dbsave = save.GetComponent<DynamicBoneSave>();
            
            for (int i = save.transform.childCount - 1; i >= 0; i--)
                GameObject.DestroyImmediate(save.transform.GetChild(i).gameObject);

            //save
            //deal with DynamicBone
            Dictionary<string, GameObject> dict = new Dictionary<string, GameObject>();
            HashSet<DynamicBoneColliderBase> dealwithedDBCs = new HashSet<DynamicBoneColliderBase>();
            DynamicBone[] oldDbs = go.GetComponents<DynamicBone>();
            if (oldDbs == null || oldDbs.Length == 0)
                return "Have not DynamicBone Script";
            foreach (DynamicBone oldDb in oldDbs)
            {
                if (oldDb.m_Root == null)
                    return "Some DynamicBone m_Root is null";
                GameObject dbGo = QueryGameObject(dict, oldDb.m_Root.name, save);
                DynamicBone dbComp = CopyCompontent<DynamicBone>(oldDb, save);
                dbComp.m_Root = dbGo.transform;
                dbComp.m_Colliders.Clear();
                //deal with DynamicBoneColliders in DynamicBone
                for (int i = 0; i < oldDb.m_Colliders.Count; i++)
                {
                    if (oldDb.m_Colliders[i] == null)
                        return "Some DynamicBone m_Colliders has null";
                    GameObject dbcGo = QueryGameObject(dict, oldDb.m_Colliders[i].name, save);
                    DynamicBoneColliderBase dbcComp = CopyCompontent<DynamicBoneColliderBase>(oldDb.m_Colliders[i], dbcGo);
                    dbComp.m_Colliders.Add(dbcComp);
                    dealwithedDBCs.Add(oldDb.m_Colliders[i]);
                }
            }

            //check other DynamicBoneCollider
            DynamicBoneColliderBase[] dbcs = go.GetComponentsInChildren<DynamicBoneColliderBase>(true);
            foreach(DynamicBoneColliderBase dbc in dbcs)
            {
                if (dealwithedDBCs.Contains(dbc))
                    continue;
                return "DynamicBoneCollider " + dbc.name + " have not be use.";
            }

            return "Save DynamicBone Script success.";
        }

        public static string Load(GameObject go, GameObject load)
        {
            //destroy old comp
            DynamicBone[] dbs = go.GetComponentsInChildren<DynamicBone>();
            foreach (DynamicBone db in dbs)
                DestroyImmediate(db);
            DynamicBoneColliderBase[] dbcs = go.GetComponentsInChildren<DynamicBoneColliderBase>();
            foreach (DynamicBoneColliderBase dbc in dbcs)
                DestroyImmediate(dbc);

            //create dict
            Dictionary<string, Transform> dict = new Dictionary<string, Transform>();
            CreateDict(go.transform, dict);

            //copy DynamicBoneColliderBase
            dbcs = load.GetComponentsInChildren<DynamicBoneColliderBase>();
            foreach (DynamicBoneColliderBase dbc in dbcs)
            {
                if (!dict.ContainsKey(dbc.name))
                {
                    ClearDynamicBoneInfo(go);
                    return "新prefab上无法找到保存的DynamicBoneCollider所在节点: " + dbc.name;
                }
                CopyCompontent<DynamicBoneColliderBase>(dbc, dict[dbc.name].gameObject);
            }

            //copy DynamicBone
            dbs = load.GetComponentsInChildren<DynamicBone>();
            foreach(DynamicBone db in dbs)
            {
                DynamicBone newDb = CopyCompontent<DynamicBone>(db, go);
                if (!dict.ContainsKey(db.m_Root.name))
                {
                    ClearDynamicBoneInfo(go);
                    return "新prefab上无法找到保存的m_Root节点: " + db.m_Root.name;
                }
                newDb.m_Root = dict[db.m_Root.name];
                newDb.m_Colliders.Clear();
                for(int i = 0; i < db.m_Colliders.Count; i++)
                {
                    if (!dict.ContainsKey(db.m_Colliders[i].name))
                    {
                        ClearDynamicBoneInfo(go);
                        return "新prefab上无法找到保存的m_Colliders节点: " + db.m_Colliders[i].name;
                    }
                    DynamicBoneColliderBase newDbc = dict[db.m_Colliders[i].name].GetComponent<DynamicBoneColliderBase>();
                    if (newDbc == null)
                    {
                        ClearDynamicBoneInfo(go);
                        return "新prefab的m_Colliders没有DynamicBoneCollider脚本，请联系pyc";
                    }
                    newDb.m_Colliders.Add(newDbc);
                }
            }

            return "Load DynamicBone Script success.";
        }

        private static void ClearDynamicBoneInfo(GameObject go)
        {
            DynamicBone[] dbs = go.GetComponentsInChildren<DynamicBone>();
            foreach (DynamicBone db in dbs)
                DestroyImmediate(db);
            DynamicBoneColliderBase[] dbcs = go.GetComponentsInChildren<DynamicBoneColliderBase>();
            foreach (DynamicBoneColliderBase dbc in dbcs)
                DestroyImmediate(dbc);
        }


        private static GameObject QueryGameObject(Dictionary<string, GameObject> dict, string name, GameObject root)
        {
            if (dict.ContainsKey(name))
                return dict[name];
            else
            {
                GameObject go = new GameObject(name);
                go.transform.parent = root.transform;
                dict[name] = go;
                return go;
            }
        }

        private static T CopyCompontent<T>(T oldComp, GameObject newGo) where T : Component
        {
            if (oldComp != null)
            {
                T newComp = newGo.AddComponent<T>();
                UnityEditorInternal.ComponentUtility.CopyComponent(oldComp);
                UnityEditorInternal.ComponentUtility.PasteComponentValues(newComp);
                return newComp;
            }
            return null;
        }

        private static void CreateDict(Transform ts, Dictionary<string, Transform> dict)
        {
            if (dict.ContainsKey(ts.name))
            {
                Debug.LogError("Prefab has the same bone node : " + ts.name);
            }
            else
            {
                dict.Add(ts.name, ts);
            }
            for (int i = 0; i < ts.childCount; i++)
            {
                CreateDict(ts.GetChild(i), dict);
            }
        }
    }
}