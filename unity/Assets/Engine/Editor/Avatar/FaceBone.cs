using CFUtilPoolLib;
using System.Collections.Generic;
using System.IO;
using XEditor;
using UnityEngine;
using UnityEditor;

public enum TransformRange
{
    MinValue,
    DefaultValue,
    MaxValue,
    MaxCount
}

public enum DirtyPos
{
    Position = 0,
    Rotation,
    Scale,
    MaxCount
}

public class BoneData
{
    public Transform Trans;
    public Vector3 Position;
    public Quaternion Rotation;
    public Vector3 Scale;
    public int DirtyFlag = 0;
    public string Name;

    public BoneData(string name)
    {
        Name = name;
    }

    public void Bind(Transform trans)
    {
        Trans = trans;
        Position = trans.localPosition;
        Rotation = trans.localRotation;
        Scale = trans.localScale;
        DirtyFlag = 0;
    }

    public void SyncData(BoneData other)
    {
        Position = other.Position;
        Rotation = other.Rotation;
        Scale = other.Scale;
    }

    public void SetDirty(DirtyPos posBit)
    {
        DirtyFlag |= (1 << (int)posBit);
    }

    public void UpdatePosX(float x)
    {
        Position.x = x;
        SetDirty(DirtyPos.Position);
    }
    public void UpdatePosY(float y)
    {
        Position.y = y;
        SetDirty(DirtyPos.Position);
    }
    public void UpdatePosZ(float z)
    {
        Position.z = z;
        SetDirty(DirtyPos.Position);
    }

    public void UpdateRotate(ref Quaternion rot)
    {
        Rotation = rot;
        SetDirty(DirtyPos.Rotation);
    }
    public void UpdateScaleX(float x)
    {
        Scale.x = x;
        SetDirty(DirtyPos.Scale);
    }
    public void UpdateScaleY(float y)
    {
        Scale.y = y;
        SetDirty(DirtyPos.Scale);
    }
    public void UpdateScaleZ(float z)
    {
        Scale.z = z;
        SetDirty(DirtyPos.Scale);
    }
    public void UpdateToTransfrom()
    {
        if (DirtyFlag == 0 || Trans == null) return;
        if ((DirtyFlag & (1 << (int)DirtyPos.Position)) != 0)
        {
            Trans.localPosition = Position;
        }
        if ((DirtyFlag & (1 << (int)DirtyPos.Rotation)) != 0)
        {
            Trans.localRotation = Rotation;
        }
        if ((DirtyFlag & (1 << (int)DirtyPos.Scale)) != 0)
        {
            Trans.localScale = Scale;
        }
        DirtyFlag = 0;
    }
}

public class BaseTransform
{
    public BoneData Bone;
    public FaceModifyType Type;

    public virtual void LoadData(Transform tf)
    {
        Bone.Bind(tf);
    }

    public virtual void UpdateWeight(float weight) { }
}

public class BoneRotationTransform : BaseTransform
{
    public Quaternion[] TransformData = new Quaternion[(int)TransformRange.MaxCount];

    public void LoadData(ref Quaternion minValue, ref Quaternion maxValue)
    {
        TransformData[(int)TransformRange.MinValue] = minValue;
        TransformData[(int)TransformRange.MaxValue] = maxValue;
    }

    public override void LoadData(Transform tf)
    {
        base.LoadData(tf);
        TransformData[(int)TransformRange.DefaultValue] = Bone.Rotation;
    }

    public override void UpdateWeight(float weight)
    {
        float absWeight = Mathf.Abs(weight);
        int id = weight < 0 ? (int)TransformRange.MinValue : (int)TransformRange.MaxValue;
        var CurrentData = Quaternion.Lerp(TransformData[(int)TransformRange.DefaultValue], TransformData[id], absWeight);
        Bone.UpdateRotate(ref CurrentData);
        base.UpdateWeight(weight);
    }
}

public class BoneTransform : BaseTransform
{
    public float[] TransformData = new float[(int)TransformRange.MaxCount];

    public void LoadData(float minValue, float maxValue)
    {
        TransformData[(int)TransformRange.MinValue] = minValue;
        TransformData[(int)TransformRange.MaxValue] = maxValue;
    }

    public override void LoadData(Transform tf)
    {
        base.LoadData(tf);
        switch (Type)
        {
            case FaceModifyType.PositionX:
                TransformData[(int)TransformRange.DefaultValue] = Bone.Position.x;
                break;
            case FaceModifyType.PositionY:
                TransformData[(int)TransformRange.DefaultValue] = Bone.Position.y;
                break;
            case FaceModifyType.PositionZ:
                TransformData[(int)TransformRange.DefaultValue] = Bone.Position.z;
                break;
            case FaceModifyType.ScaleX:
                TransformData[(int)TransformRange.DefaultValue] = Bone.Scale.x;
                break;
            case FaceModifyType.ScaleY:
                TransformData[(int)TransformRange.DefaultValue] = Bone.Scale.y;
                break;
            case FaceModifyType.ScaleZ:
                TransformData[(int)TransformRange.DefaultValue] = Bone.Scale.z;
                break;
        }
    }

    public override void UpdateWeight(float weight)
    {
        float absWeight = Mathf.Abs(weight);
        int id = weight < 0 ? (int)TransformRange.MinValue : (int)TransformRange.MaxValue;
        var CurrentData = Mathf.Lerp(TransformData[(int)TransformRange.DefaultValue], TransformData[id], absWeight);
        //XDebug.singleton.AddGreenLog("min: " + TransformData[(int)TransformRange.DefaultValue] + " max: " + TransformData[id]+ " weight: " + absWeight + " curr: " + CurrentData + " tf: " + Bone.Trans.name + " type: " + Type);
        switch (Type)
        {
            case FaceModifyType.PositionX:
                Bone.UpdatePosX(CurrentData);
                break;
            case FaceModifyType.PositionY:
                Bone.UpdatePosY(CurrentData);
                break;
            case FaceModifyType.PositionZ:
                Bone.UpdatePosZ(CurrentData);
                break;
            case FaceModifyType.ScaleX:
                Bone.UpdateScaleX(CurrentData);
                break;
            case FaceModifyType.ScaleY:
                Bone.UpdateScaleY(CurrentData);
                break;
            case FaceModifyType.ScaleZ:
                Bone.UpdateScaleZ(CurrentData);
                break;
        }
        base.UpdateWeight(weight);
    }
}

public class FaceBone
{

    public XRoleParts RoleParts;
    private List<List<BaseTransform>> controlGroups = null;
    private Dictionary<Transform, List<BoneData>> tfKneadFace = new Dictionary<Transform, List<BoneData>>();
    private FaceData data;
    private Vector2 rect;

    public FaceBone(FaceData dt)
    {
        data = dt;
        Bind();
    }

    public void Load(GameObject go, RoleShape shape)
    {
        RoleParts = go.GetComponent<XRoleParts>();
        string path = "Assets/BundleRes/Config/" + shape.ToString().ToLower();
        TextAsset ta = AssetDatabase.LoadAssetAtPath<TextAsset>(path + ".bytes");
        if (ta != null)
        {
            MemoryStream ms = new MemoryStream(ta.bytes);
            FaceBoneDatas fdata = new FaceBoneDatas(ms);
            MakeControlGroups(fdata);
            ms.Close();
        }
    }

    private int SearchKnead(string name)
    {
        var childs = RoleParts.knead;
        for (int i = 0; i < childs.Length; i++)
        {
            if (childs[i].name.Equals(name))
            {
                return i;
            }
        }
        return -1;
    }


    private void MakeControlGroups(FaceBoneDatas fdata)
    {
        if (controlGroups == null)
        {
            int len = fdata.BoneDatas.Length;
            var bones = new BaseTransform[len];
            for (var i = 0; i < len; i++)
            {
                string name = fdata.BoneDatas[i].name;
                FaceModifyType type = fdata.BoneDatas[i].type;
                BoneData data = new BoneData(name);
                BaseTransform bt;
                if (type == FaceModifyType.Rotation)
                {
                    bt = new BoneRotationTransform() { Bone = data, Type = type };
                    (bt as BoneRotationTransform).LoadData(ref fdata.BoneDatas[i].minRot, ref fdata.BoneDatas[i].maxRot);
                }
                else
                {
                    bt = new BoneTransform() { Bone = data, Type = type };
                    (bt as BoneTransform).LoadData(fdata.BoneDatas[i].minValue, fdata.BoneDatas[i].maxValue);
                }
                int find = SearchKnead(name);
                if (find >= 0)
                {
                    Transform tf = RoleParts.knead[find];
                    if (tf == null) XDebug.singleton.AddErrorLog("face bone is null ", name);
                    bt.LoadData(tf);
                    if (tfKneadFace.ContainsKey(tf))
                    {
                        tfKneadFace[tf].Add(bt.Bone);
                    }
                    else
                    {
                        List<BoneData> list = new List<BoneData>();
                        list.Add(bt.Bone);
                        tfKneadFace[tf] = list;
                    }
                }
                else XDebug.singleton.AddErrorLog("not found tranf: ", name);
                bones[i] = bt;
            }
            var groupCount = fdata.Groups.Length;
            controlGroups = new List<List<BaseTransform>>(groupCount);
            for (var k = 0; k < groupCount; k++)
            {
                var controlCount = fdata.Groups[k].controlCount;
                var controls = new List<BaseTransform>(controlCount);
                for (var i = 0; i < controlCount; i++)
                {
                    short controlId = fdata.Groups[k].controlIds[i];
                    controls.Add(bones[controlId]);
                }
                controlGroups.Add(controls);
            }
        }
    }

    private bool[] folds;
    private Object[] icons;
    private float[] args;
    string prefix = @"Assets/BundleRes/Faceicon/";

    private void Bind()
    {
        int cnt = 0;
        for (int i = 0; i < data.headData.Length; i++)
        {
            cnt += BindItem(data.headData[i]);
        }
        for (int i = 0; i < data.senseData.Length; i++)
        {
            cnt += BindItem(data.senseData[i]);
        }
        args = new float[cnt];
        for (int i = 0; i < cnt; i++) args[i] = 0.5f;
    }

    private int BindItem(FaceBaseData data)
    {
        int cnt = 0;
        if (data.v2Type != FaceV2Type.None) cnt += 2;
        if (data.values != null) cnt += data.values.Length;
        return cnt;
    }

    public void OnGui()
    {
        GUILayout.Space(16);
        int cnt = data.headData.Length + data.senseData.Length;
        if (folds == null) folds = new bool[cnt];
        if (icons == null) icons = new Object[cnt];
        rect = GUILayout.BeginScrollView(rect);
        int j = 0;
        for (int i = 0; i < data.headData.Length; i++)
        {
            GuiItem(data.headData[i], i, ref j);
        }
        for (int i = 0; i < data.senseData.Length; i++)
        {
            GuiItem(data.senseData[i], i + data.headData.Length, ref j);
        }
        GUILayout.EndScrollView();
    }



    private void GuiItem(FaceBaseData data, int ix, ref int jx)
    {
        folds[ix] = EditorGUILayout.Foldout(folds[ix], data.name);
        if (folds[ix])
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.BeginVertical();
            if (data.v2Type != FaceV2Type.None)
            {
                int idx = (int)data.v2Type;
                string name = XEditorUtil.Config.facev2Type[idx];
                GuiSlider(name + "X", ref jx);
                GuiSlider(name + "Y", ref jx);
            }
            else if (data.values != null)
            {
                for (int i = 0; i < data.values.Length; i++)
                {
                    FaceValueType type = data.values[i];
                    string name = XEditorUtil.Config.faceType[(int)type];
                    float v = 0.5f;
                    GuiSlider(name, ref jx);
                }
            }
            EditorGUILayout.EndVertical();
            EditorGUILayout.Space();
            if (!string.IsNullOrEmpty(data.icon) && icons[ix] == null)
                icons[ix] = AssetDatabase.LoadAssetAtPath<Texture>(prefix + data.icon + ".png");
            EditorGUILayout.ObjectField(icons[ix], typeof(Texture), true, GUILayout.Width(56), GUILayout.Height(56));
            EditorGUILayout.EndHorizontal();
        }
    }

    private void GuiSlider(string name, ref int ix)
    {
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField("    " + name);
        args[ix] = EditorGUILayout.Slider(args[ix], 0, 1);
        EditorGUILayout.Space();
        EditorGUILayout.EndHorizontal();
        ix++;
    }

    private void ProcessKneadBone(int groupId, float weight)
    {
        if (controlGroups == null || groupId >= controlGroups.Count) return;
        var controls = controlGroups[groupId];
        for (int i = 0, count = controls.Count; i < count; ++i)
        {
            controls[i].UpdateWeight(weight);
            SyncSameTfControlsData(controls[i].Bone);
        }
        for (int i = 0, count = controls.Count; i < count; ++i)
        {
            controls[i].Bone.UpdateToTransfrom();
        }
    }


    private void SyncSameTfControlsData(BoneData bone)
    {
        Transform tf = bone.Trans;
        if (tfKneadFace.Count <= 0 || !tfKneadFace.ContainsKey(tf)) return;
        var list = tfKneadFace[tf];
        for (int i = 0; i < list.Count; i++)
        {
            if (list[i] != bone)
            {
                list[i].SyncData(bone);
            }
        }
    }

}