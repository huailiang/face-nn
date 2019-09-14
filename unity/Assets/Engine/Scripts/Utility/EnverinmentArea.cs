#if UNITY_EDITOR
using System;
using System.Collections.Generic;

using CFUtilPoolLib;
using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;
using UnityEngine.Rendering;
namespace CFEngine
{

    [Serializable]
    public class EnvBox
    {
        [System.NonSerialized]
        public BoxBoundsHandle boundsHandle = new BoxBoundsHandle();
        public Vector3 center;
        public Vector3 size;
        public float rotY;

    }

    [Serializable]
    public class SceneEnvWrapper
    {
        public virtual bool IsValid()
        {
            return false;
        }
        public virtual void Valid(bool valid)
        { }

        public virtual IEnverimnentModify GetEnverimnent()
        {
            return null;
        }
    }

    [Serializable]
    public class EnvWrapper
    {
        public IEnverimnentLerp envLerp;
        public bool valid = false;

        public virtual void OnGUI()
        {

        }

        public virtual void Update()
        {

        }
        public virtual SceneEnvWrapper Fill()
        {
            return null;
        }
        public virtual void Load(SceneEnvWrapper wrapper)
        {

        }

    }

    [System.Serializable]
    public class SceneLightWrapper : SceneEnvWrapper
    {
        public LightingModify envModify;
        public bool isValid = false;
        public string roleLight0Path;
        public string roleLight1Path;
        public string sceneRuntimeLight0Path;
        public string sceneRuntimeLight1Path;
        public override bool IsValid()
        {
            return isValid;
        }
        public override void Valid(bool valid)
        {
            isValid = valid;
        }
        public override IEnverimnentModify GetEnverimnent()
        {
            return envModify;
        }
    }

    [Serializable]
    public class LightWrapper : EnvWrapper
    {
        public LightingModify envModify = new LightingModify();

        public Light roleLight0;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight0Rot;

        public Light roleLight1;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight1Rot;

        public Light sceneRuntimeLight0;

        [System.NonSerialized]
        public TransformRotationGUIWrapper sceneRuntimeLight0Rot;

        public Light sceneRuntimeLight1;

        [System.NonSerialized]
        public TransformRotationGUIWrapper sceneRuntimeLight1Rot;

        void LightInstpetorGui(string name, ref Light light, ref TransformRotationGUIWrapper wrapper)
        {
            light = EditorGUILayout.ObjectField(name, light, typeof(Light), true) as Light;

            if (light != null)
            {
                light.color = EditorGUILayout.ColorField("Color", light.color);
                light.intensity = EditorGUILayout.Slider("Intensity", light.intensity, 0, 10.0f);
                if (wrapper != null)
                    wrapper.OnGUI();
            }
        }
        void PrepareLightInspector(Light light, ref TransformRotationGUIWrapper rot)
        {
            if (light != null)
            {
                if (rot == null || rot.t != light.transform)
                {
                    rot = EditorCommon.GetTransformRotatGUI(light.transform);
                }
            }
        }

        void SyncLight(Light light, ref TransformRotationGUIWrapper wrapper, ref LightInfo lightInfo)
        {
            if (light != null)
            {
                lightInfo.lightColor = light.color;
                lightInfo.lightDir = light.transform.rotation * -Vector3.forward;
                lightInfo.lightDir.w = light.intensity;
            }
            else
            {
                lightInfo.lightDir.w = 0;
                lightInfo.lightColor.a = 0;
            }

        }
        private void FindLightObject(string path, ref Light light)
        {
            if (!string.IsNullOrEmpty(path))
            {
                GameObject go = GameObject.Find(path);
                if (go != null)
                {
                    light = go.GetComponent<Light>();
                }
            }
        }
        public override void OnGUI()
        {
            PrepareLightInspector(roleLight0, ref roleLight0Rot);
            LightInstpetorGui("RoleLight0", ref roleLight0, ref roleLight0Rot);
            PrepareLightInspector(roleLight1, ref roleLight1Rot);
            LightInstpetorGui("RoleLight1", ref roleLight1, ref roleLight1Rot);
            PrepareLightInspector(sceneRuntimeLight0, ref sceneRuntimeLight0Rot);
            LightInstpetorGui("SceneLight0", ref sceneRuntimeLight0, ref sceneRuntimeLight0Rot);
            PrepareLightInspector(sceneRuntimeLight1, ref sceneRuntimeLight1Rot);
            LightInstpetorGui("SceneLight1", ref sceneRuntimeLight1, ref sceneRuntimeLight1Rot);
        }

        public override void Update()
        {
            SyncLight(roleLight0, ref roleLight0Rot, ref envModify.roleLightInfo0);
            SyncLight(roleLight1, ref roleLight1Rot, ref envModify.roleLightInfo1);
            SyncLight(sceneRuntimeLight0, ref sceneRuntimeLight0Rot, ref envModify.sceneLightInfo0);
            SyncLight(sceneRuntimeLight1, ref sceneRuntimeLight1Rot, ref envModify.sceneLightInfo1);
        }
        public override SceneEnvWrapper Fill()
        {
            SceneLightWrapper wrapper = new SceneLightWrapper();
            wrapper.Valid(valid);
            Update();
            wrapper.envModify = envModify;
            if (roleLight0 != null)
                wrapper.roleLight0Path = EditorCommon.GetSceneObjectPath(roleLight0.transform);
            if (roleLight1 != null)
                wrapper.roleLight1Path = EditorCommon.GetSceneObjectPath(roleLight1.transform);
            if (sceneRuntimeLight0 != null)
                wrapper.sceneRuntimeLight0Path = EditorCommon.GetSceneObjectPath(sceneRuntimeLight0.transform);
            if (sceneRuntimeLight1 != null)
                wrapper.sceneRuntimeLight1Path = EditorCommon.GetSceneObjectPath(sceneRuntimeLight1.transform);
            return wrapper;

        }

        public override void Load(SceneEnvWrapper wrapper)
        {
            SceneLightWrapper lightWrapper = wrapper as SceneLightWrapper;

            envModify.roleLightInfo0 = lightWrapper.envModify.roleLightInfo0;
            envModify.roleLightInfo1 = lightWrapper.envModify.roleLightInfo1;
            envModify.sceneLightInfo0 = lightWrapper.envModify.sceneLightInfo0;
            envModify.sceneLightInfo1 = lightWrapper.envModify.sceneLightInfo1;
            valid = lightWrapper.IsValid();
            FindLightObject(lightWrapper.roleLight0Path, ref roleLight0);
            FindLightObject(lightWrapper.roleLight1Path, ref roleLight1);
            FindLightObject(lightWrapper.sceneRuntimeLight0Path, ref sceneRuntimeLight0);
            FindLightObject(lightWrapper.sceneRuntimeLight1Path, ref sceneRuntimeLight1);
        }
    }

    [System.Serializable]
    public class SceneAmbientWrapper : SceneEnvWrapper
    {
        public AmbientModify envModify;
        public bool isValid = false;
        public override bool IsValid()
        {
            return isValid;
        }
        public override void Valid(bool valid)
        {
            isValid = valid;
        }
        public override IEnverimnentModify GetEnverimnent()
        {
            return envModify;
        }
    }

    [Serializable]
    public class AmbientWrapper : EnvWrapper
    {
        public AmbientModify envModify = new AmbientModify();
        private Material skyMat = null;
        public override void OnGUI()
        {
            envModify.ambientSkyColor = EditorGUILayout.ColorField("SkyColor", envModify.ambientSkyColor);
            if (RenderSettings.ambientMode == AmbientMode.Trilight)
            {
                envModify.ambientEquatorColor = EditorGUILayout.ColorField("EquatorColor", envModify.ambientEquatorColor);
                envModify.ambientGroundColor = EditorGUILayout.ColorField("GroundColor", envModify.ambientGroundColor);
            }
            envModify.AmbientMax = EditorGUILayout.FloatField("AmbientMax", envModify.AmbientMax);

            if (!string.IsNullOrEmpty(envModify.SkyboxMatPath))
            {
                string path = string.Format("{0}/{1}.mat", AssetsConfig.GlobalAssetsConfig.ResourcePath, envModify.SkyboxMatPath);
                skyMat = AssetDatabase.LoadAssetAtPath<Material>(path);
            }
            Material sky = EditorGUILayout.ObjectField(skyMat, typeof(Material), false) as Material;
            if (sky != skyMat)
            {
                skyMat = sky;
                if (skyMat == null)
                {
                    envModify.SkyboxMatPath = "";
                }
                else
                {
                    string path = AssetDatabase.GetAssetPath(skyMat);
                    envModify.SkyboxMatPath = path.Substring(AssetsConfig.GlobalAssetsConfig.ResourcePath.Length + 1);
                    envModify.SkyboxMatPath = envModify.SkyboxMatPath.Replace(".mat", "");
                }
            }
        }

        public override SceneEnvWrapper Fill()
        {
            SceneAmbientWrapper wrapper = new SceneAmbientWrapper();
            wrapper.Valid(valid);
            wrapper.envModify = envModify;
            return wrapper;
        }

        public override void Load(SceneEnvWrapper wrapper)
        {
            SceneAmbientWrapper ambientWrapper = wrapper as SceneAmbientWrapper;

            envModify.ambientMode = ambientWrapper.envModify.ambientMode;
            envModify.ambientSkyColor = ambientWrapper.envModify.ambientSkyColor;
            envModify.ambientEquatorColor = ambientWrapper.envModify.ambientEquatorColor;
            envModify.ambientGroundColor = ambientWrapper.envModify.ambientGroundColor;
            envModify.AmbientMax = ambientWrapper.envModify.AmbientMax;

            valid = ambientWrapper.IsValid();
        }
    }

    [System.Serializable]
    public class SceneFogWrapper : SceneEnvWrapper
    {
        public FogModify envModify;

        public bool isValid = false;
        public override bool IsValid()
        {
            return isValid;
        }
        public override void Valid(bool valid)
        {
            isValid = valid;
        }
        public override IEnverimnentModify GetEnverimnent()
        {
            return envModify;
        }
    }

    [Serializable]
    public class FogWrapper : EnvWrapper
    {
        public FogModify envModify = new FogModify();

        public override void OnGUI()
        {
            envModify.Density = EditorGUILayout.FloatField("Density", envModify.Density);
            envModify.EndHeight = EditorGUILayout.FloatField("EndHeight", envModify.EndHeight);
            envModify.StartDistance = EditorGUILayout.FloatField("StartDistance", envModify.StartDistance);
            envModify.SkyboxHeight = EditorGUILayout.FloatField("SkyboxHeight", envModify.SkyboxHeight);
            envModify.Color0 = EditorGUILayout.ColorField("Color0", envModify.Color0);
            envModify.Color1 = EditorGUILayout.ColorField("Color1", envModify.Color1);
            envModify.Color2 = EditorGUILayout.ColorField("Color2", envModify.Color2);
        }

        public override SceneEnvWrapper Fill()
        {
            SceneFogWrapper wrapper = new SceneFogWrapper();
            wrapper.Valid(valid);
            wrapper.envModify = envModify;
            return wrapper;
        }

        public override void Load(SceneEnvWrapper wrapper)
        {
            SceneFogWrapper fogWrapper = wrapper as SceneFogWrapper;

            envModify.Density = fogWrapper.envModify.Density;
            envModify.EndHeight = fogWrapper.envModify.EndHeight;
            envModify.StartDistance = fogWrapper.envModify.StartDistance;
            envModify.SkyboxHeight = fogWrapper.envModify.SkyboxHeight;
            envModify.Color0 = fogWrapper.envModify.Color0;
            envModify.Color1 = fogWrapper.envModify.Color1;
            envModify.Color2 = fogWrapper.envModify.Color2;
            valid = fogWrapper.IsValid();
        }
    }

    [System.Serializable]
    public class SceneBloomWrapper : SceneEnvWrapper
    {
        public BloomModify envModify;

        public bool isValid = false;
        public override bool IsValid()
        {
            return isValid;
        }
        public override void Valid(bool valid)
        {
            isValid = valid;
        }
        public override IEnverimnentModify GetEnverimnent()
        {
            return envModify;
        }
    }

    [Serializable]
    public class BloomWrapper : EnvWrapper
    {
        public BloomModify envModify = new BloomModify();

        public override void OnGUI()
        {
            envModify.enabled = EditorGUILayout.Toggle("enabled", envModify.enabled);
            envModify.intensity = EditorGUILayout.FloatField("intensity", envModify.intensity);
            envModify.threshold = EditorGUILayout.FloatField("threshold", envModify.threshold);
            envModify.softKnee = EditorGUILayout.FloatField("softKnee", envModify.softKnee);
            envModify.diffusion = EditorGUILayout.FloatField("diffusion", envModify.diffusion);
            envModify.color = EditorGUILayout.ColorField("color", envModify.color);
        }

        public override SceneEnvWrapper Fill()
        {
            SceneBloomWrapper wrapper = new SceneBloomWrapper();
            wrapper.Valid(valid);
            wrapper.envModify = envModify;
            return wrapper;
        }

        public override void Load(SceneEnvWrapper wrapper)
        {
            SceneBloomWrapper bloomWrapper = wrapper as SceneBloomWrapper;

            envModify.enabled = bloomWrapper.envModify.enabled;
            envModify.intensity = bloomWrapper.envModify.intensity;
            envModify.threshold = bloomWrapper.envModify.threshold;
            envModify.softKnee = bloomWrapper.envModify.softKnee;
            envModify.diffusion = bloomWrapper.envModify.diffusion;
            envModify.color = bloomWrapper.envModify.color;
            valid = bloomWrapper.IsValid();
        }
    }

    [System.Serializable]
    public class SceneLutWrapper : SceneEnvWrapper
    {
        public LutModify envModify;

        public bool isValid = false;
        public override bool IsValid()
        {
            return isValid;
        }
        public override void Valid(bool valid)
        {
            isValid = valid;
        }
        public override IEnverimnentModify GetEnverimnent()
        {
            return envModify;
        }
    }

    [Serializable]
    public class LutWrapper : EnvWrapper
    {
        public LutModify envModify = new LutModify();
        public override void OnGUI()
        {
            envModify.temperature = EditorGUILayout.FloatField("temperature", envModify.temperature);
            envModify.tint = EditorGUILayout.FloatField("threshold", envModify.tint);
            envModify.colorFilter = EditorGUILayout.ColorField("colorFilter", envModify.colorFilter);
            envModify.hueShift = EditorGUILayout.FloatField("hueShift", envModify.hueShift);
            envModify.saturation = EditorGUILayout.FloatField("saturation", envModify.saturation);
            envModify.postExposure = EditorGUILayout.FloatField("postExposure", envModify.postExposure);
            envModify.contrast = EditorGUILayout.FloatField("contrast", envModify.contrast);
            envModify.mixerRedOutRedIn = EditorGUILayout.FloatField("mixerRedOutRedIn", envModify.mixerRedOutRedIn);
            envModify.mixerRedOutGreenIn = EditorGUILayout.FloatField("mixerRedOutGreenIn", envModify.mixerRedOutGreenIn);
            envModify.mixerRedOutBlueIn = EditorGUILayout.FloatField("mixerRedOutBlueIn", envModify.mixerRedOutBlueIn);
            envModify.mixerGreenOutRedIn = EditorGUILayout.FloatField("mixerGreenOutRedIn", envModify.mixerGreenOutRedIn);
            envModify.mixerGreenOutGreenIn = EditorGUILayout.FloatField("mixerGreenOutGreenIn", envModify.mixerGreenOutGreenIn);
            envModify.mixerGreenOutBlueIn = EditorGUILayout.FloatField("mixerGreenOutBlueIn", envModify.mixerGreenOutBlueIn);
            envModify.mixerBlueOutRedIn = EditorGUILayout.FloatField("mixerBlueOutRedIn", envModify.mixerBlueOutRedIn);
            envModify.mixerBlueOutGreenIn = EditorGUILayout.FloatField("mixerBlueOutGreenIn", envModify.mixerBlueOutGreenIn);
            envModify.mixerBlueOutBlueIn = EditorGUILayout.FloatField("mixerBlueOutBlueIn", envModify.mixerBlueOutBlueIn);
            envModify.lift = EditorGUILayout.Vector4Field("lift", envModify.lift);
            envModify.gamma = EditorGUILayout.Vector4Field("gamma", envModify.gamma);
            envModify.gain = EditorGUILayout.Vector4Field("gain", envModify.gain);
        }

        public override SceneEnvWrapper Fill()
        {
            SceneLutWrapper wrapper = new SceneLutWrapper();
            wrapper.Valid(valid);
            wrapper.envModify = envModify;
            return wrapper;
        }

        public override void Load(SceneEnvWrapper wrapper)
        {
            SceneLutWrapper lutWrapper = wrapper as SceneLutWrapper;

            envModify.enabled = lutWrapper.envModify.enabled;
            envModify.toneCurveToeStrength = lutWrapper.envModify.toneCurveToeStrength;
            envModify.toneCurveToeLength = lutWrapper.envModify.toneCurveToeLength;
            envModify.toneCurveShoulderStrength = lutWrapper.envModify.toneCurveShoulderStrength;
            envModify.toneCurveShoulderLength = lutWrapper.envModify.toneCurveShoulderLength;
            envModify.toneCurveShoulderAngle = lutWrapper.envModify.toneCurveShoulderAngle;
            envModify.toneCurveGamma = lutWrapper.envModify.toneCurveGamma;

            envModify.temperature = lutWrapper.envModify.temperature;
            envModify.tint = lutWrapper.envModify.tint;
            envModify.colorFilter = lutWrapper.envModify.colorFilter;
            envModify.hueShift = lutWrapper.envModify.hueShift;
            envModify.saturation = lutWrapper.envModify.saturation;
            envModify.postExposure = lutWrapper.envModify.postExposure;
            envModify.contrast = lutWrapper.envModify.contrast;
            envModify.mixerRedOutRedIn = lutWrapper.envModify.mixerRedOutRedIn;
            envModify.mixerRedOutGreenIn = lutWrapper.envModify.mixerRedOutGreenIn;
            envModify.mixerRedOutBlueIn = lutWrapper.envModify.mixerRedOutBlueIn;
            envModify.mixerGreenOutRedIn = lutWrapper.envModify.mixerGreenOutRedIn;
            envModify.mixerGreenOutGreenIn = lutWrapper.envModify.mixerGreenOutGreenIn;
            envModify.mixerGreenOutBlueIn = lutWrapper.envModify.mixerGreenOutBlueIn;
            envModify.mixerBlueOutRedIn = lutWrapper.envModify.mixerBlueOutRedIn;
            envModify.mixerBlueOutGreenIn = lutWrapper.envModify.mixerBlueOutGreenIn;
            envModify.mixerBlueOutBlueIn = lutWrapper.envModify.mixerBlueOutBlueIn;
            envModify.lift = lutWrapper.envModify.lift;
            envModify.gamma = lutWrapper.envModify.gamma;
            envModify.gain = lutWrapper.envModify.gain;

            valid = lutWrapper.IsValid();
        }
    }

    [System.Serializable]
    public class SceneEffectWrapper : SceneEnvWrapper
    {
        public EffectModify envModify;

        public bool isValid = false;
        public override bool IsValid()
        {
            return isValid;
        }
        public override void Valid(bool valid)
        {
            isValid = valid;
        }
        public override IEnverimnentModify GetEnverimnent()
        {
            return envModify;
        }
    }

    [Serializable]
    public class EffectWrapper : EnvWrapper
    {
        public EffectModify envModify = new EffectModify();

        public override void OnGUI()
        {
            envModify.areaId = (ushort)EditorGUILayout.IntField("areaId", envModify.areaId);
            envModify.effectType = (short)EditorGUILayout.IntField("effectType", envModify.effectType);
            envModify.triggerCount = (short)EditorGUILayout.IntField("triggerCount", envModify.triggerCount);
        }

        public override SceneEnvWrapper Fill()
        {
            SceneEffectWrapper wrapper = new SceneEffectWrapper();
            wrapper.Valid(valid);
            wrapper.envModify = envModify;
            return wrapper;
        }

        public override void Load(SceneEnvWrapper wrapper)
        {
            SceneEffectWrapper effectWrapper = wrapper as SceneEffectWrapper;

            envModify.areaId = effectWrapper.envModify.areaId;
            envModify.effectType = effectWrapper.envModify.effectType;
            envModify.triggerCount = effectWrapper.envModify.triggerCount;
            valid = effectWrapper.IsValid();
        }
    }

    [DisallowMultipleComponent, ExecuteInEditMode]
    public class EnverinmentArea : MonoBehaviour
    {
        public string dynamicSceneName;
         [CFRange(-1,10)]
        public float lerpTime = 1.0f;
        public List<EnvBox> areaList = new List<EnvBox>();
        public Color color = Color.red;
        public bool active = false;
        [NonSerialized]
        public float blinkTime = 0;
        [NonSerialized]
        public EnvWrapper[] envModifyList = new EnvWrapper[(int)EnverimentModifyType.Num];

        public LightWrapper lightingWrapper = new LightWrapper();
        public AmbientWrapper ambientWrapper = new AmbientWrapper();
        public WeatherModify weatherModify = new WeatherModify();
        public FogWrapper fogWrapper = new FogWrapper();
        public BloomWrapper bloomWrapper = new BloomWrapper();
        public LutWrapper lutWrapper = new LutWrapper();
        public VolumnLightModify volumnLightModify = new VolumnLightModify();
        public VignetteModify vignetteModify = new VignetteModify();
        public EffectWrapper effectModify = new EffectWrapper();
        public bool areaFolder = true;
        public bool envModifyFolder = true;
        void Update()
        {
            FillData();

        }
        public void FillData()
        {
            if (envModifyList[(int)EnverimentModifyType.Lighting] == null)
            {
                lightingWrapper.envLerp = lightingWrapper.envModify;
                envModifyList[(int)EnverimentModifyType.Lighting] = lightingWrapper;
            }

            if (envModifyList[(int)EnverimentModifyType.Ambient] == null)
            {
                ambientWrapper.envLerp = ambientWrapper.envModify;
                envModifyList[(int)EnverimentModifyType.Ambient] = ambientWrapper;
            }
            if (envModifyList[(int)EnverimentModifyType.Fog] == null)
            {
                fogWrapper.envLerp = fogWrapper.envModify;
                envModifyList[(int)EnverimentModifyType.Fog] = fogWrapper;
            }
            if (envModifyList[(int)EnverimentModifyType.PPBloom] == null)
            {
                bloomWrapper.envLerp = bloomWrapper.envModify;
                envModifyList[(int)EnverimentModifyType.PPBloom] = bloomWrapper;
            }
            if (envModifyList[(int)EnverimentModifyType.PPLut] == null)
            {
                lutWrapper.envLerp = lutWrapper.envModify;
                envModifyList[(int)EnverimentModifyType.PPLut] = lutWrapper;
            }
            if (envModifyList[(int)EnverimentModifyType.Effect] == null)
            {
                envModifyList[(int)EnverimentModifyType.Effect] = effectModify;
            }

        }
        private static void UpdateAreaModify(EnverinmentArea area)
        {
            for (int k = 0; k < area.envModifyList.Length; ++k)
            {
                EnvWrapper wrapper = area.envModifyList[k];
                if (wrapper != null && wrapper.valid)
                {
                    wrapper.Update();
                    // envModifyList.Add (wrapper.envLerp as IEnverimnentModify);
                }
            }
        }
     

        public void Save()
        {

        }

        public void Load()
        {

        }

    }
}
#endif
