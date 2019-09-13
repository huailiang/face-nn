using System.Collections.Generic;
using UnityEngine;

namespace XEditor
{
    public class XBoneTree
    {
        /// <summary>
        /// 完整路径名
        /// </summary>
        public string cname;
        /// <summary>
        /// 名字
        /// </summary>
        public string name;
        /// <summary>
        /// 深度
        /// </summary>
        public int depth;
        /// <summary>
        /// 是否选中
        /// </summary>
        public bool select;
        /// <summary>
        /// 孩子节点
        /// </summary>
        public XBoneTree[] childs;

        private static List<string> list = new List<string>();


        public XBoneTree(string _root, string _name, int _depth)
        {
            cname = string.IsNullOrEmpty(_root) ? _name : _root + "/" + _name;
            name = _name;
            depth = _depth;
            select = false;
        }

        public void FillChilds(Transform transf)
        {
            int cnt = transf.childCount;
            childs = new XBoneTree[cnt];
            for (int i = 0; i < cnt; i++)
            {
                Transform child = transf.GetChild(i);
                childs[i] = new XBoneTree(cname, child.name, depth + 1);
                childs[i].FillChilds(child);
            }
        }

        public Vector2 GUI(Vector2 pos)
        {
            var ret = GUILayout.BeginScrollView(pos);
            GUI();
            GUILayout.EndScrollView();
            return ret;
        }

        private void GUI()
        {
            GUILayout.Toggle(select, GetSpace(depth) + name);
            if (childs != null)
            {
                for (int i = 0; i < childs.Length; i++)
                {
                    childs[i].GUI();
                }
            }
        }

        public XBoneTree SearchTree(string name)
        {
            if (this.name == name) return this;
            if (childs != null)
            {
                for (int i = 0; i < childs.Length; i++)
                {
                    var tree = childs[i].SearchTree(name);
                    if (tree != null) return tree;
                }
            }
            return null;
        }

        public Transform SearchChild(string name, GameObject root)
        {
            XBoneTree child = SearchTree(name);
            if (child != null)
            {
                int index = child.cname.IndexOf("/");
                string path = child.cname.Substring(index + 1);
                return root.transform.Find(path);
            }
            return null;
        }

        public void SetSelect(string bone, bool select)
        {
            if (name == bone) this.select = select;
            if (childs != null)
            {
                for (int i = 0; i < childs.Length; i++)
                {
                    childs[i].SetSelect(bone, select);
                }
            }
        }

        public string[] GetSelects()
        {
            list.Clear();
            AddSelect();
            return list.ToArray();
        }

        public void AddSelect()
        {
            if (select) list.Add(name);
            if (childs != null)
            {
                for (int i = 0; i < childs.Length; i++)
                {
                    childs[i].AddSelect();
                }
            }
        }

        private string GetSpace(int deep)
        {
            string ret = string.Empty;
            switch (deep)
            {
                case 0:
                    ret = "";
                    break;
                case 1:
                    ret = "  ";
                    break;
                case 2:
                    ret = "    ";
                    break;
                case 3:
                    ret = "      ";
                    break;
                case 4:
                    ret = "        ";
                    break;
                case 5:
                    ret = "          ";
                    break;
                case 6:
                    ret = "            ";
                    break;
                case 7:
                    ret = "              ";
                    break;
                case 8:
                    ret = "                ";
                    break;
                case 9:
                    ret = "                  ";
                    break;
                case 10:
                    ret = "                    ";
                    break;
                case 11:
                    ret = "                      ";
                    break;
                case 12:
                    ret = "                        ";
                    break;
                case 13:
                    ret = "                          ";
                    break;
                default:
                    ret = "                            ";
                    break;
            }
            return ret;
        }

    }
}