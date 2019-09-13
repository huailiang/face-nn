using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// 对动态骨骼的扩展 
/// **  换装的时候用到 **
/// </summary>


public partial class DynamicBone
{

    public List<string> m_strColliders = null;


    void Init()
    {
        Bind();
    }


    public void Bind()
    {
        if (m_strColliders != null)
        {
            int len = m_strColliders.Count;
            Transform root = SearchRoot();
            if (m_Colliders == null)
            {
                m_Colliders = new List<DynamicBoneColliderBase>();
            }
            else
            {
                m_Colliders.Clear();
            }
            for (int i = 0; i < len; i++)
            {
                var child = root.Find(m_strColliders[i]);
                if (child != null)
                {
                    var bc = child.GetComponent<DynamicBoneColliderBase>();
                    m_Colliders.Add(bc);
                }
            }
        }
    }


    public Transform SearchRoot()
    {
        Transform tf = transform;
        while (tf.parent != null)
        {
            bool jump = tf.name == "root";
            tf = tf.parent;
            if (jump)
            {
                break;
            }
        }
        return tf;
    }

    public static bool isValid(ref Vector3 v)
    {
        return !float.IsNaN(v.x) && !float.IsNaN(v.y) && !float.IsNaN(v.z);
    }

}
