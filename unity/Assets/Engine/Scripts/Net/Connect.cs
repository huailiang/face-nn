using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace XEngine
{

    public class ParamMessage
    {
        public RoleShape shape;
        public float[] param;
        public string name;
    }

    public class Connect
    {
        UdpClient udp;
        IPEndPoint point;
        Thread thread;
        bool connected = false;

        public bool Connected { get { return connected; } }

        Queue<ParamMessage> messages = new Queue<ParamMessage>();

        public void Initial(int port)
        {
            udp = new UdpClient(port);
            point = new IPEndPoint(IPAddress.Any, port);
            thread = new Thread(Receive);
            connected = true;
            thread.IsBackground = true;
            messages.Clear();
            thread.Start();
            Debug.Log("initial success send port:" + port);
        }


        public void Receive()
        {
            while (true)
            {
                byte[] recivcedata = udp.Receive(ref point);
                string str = Encoding.ASCII.GetString(recivcedata, 0, recivcedata.Length);
                char head = str[0];
                string body = str.Substring(1);
                if (head.Equals('q')) // quit
                {
                    break;
                }
                else if (head == 'p')
                {
                    ParamMessage msg = JsonUtility.FromJson<ParamMessage>(body);
                    Monitor.Enter(messages);
                    if (messages.Count < 1024)
                        messages.Enqueue(msg);
                    Monitor.Exit(messages);
                }
                else if (head == 'm')
                {
                    Debug.Log(body);
                }
            }
            Quit();
        }


        public void Quit()
        {
            Debug.Log("unity connect close");
            if (udp != null)
            {
                udp.Close();
            }
            if (thread != null)
            {
                thread.Abort();
            }
            messages.Clear();
            connected = false;
        }

        public ParamMessage FetchMessage()
        {
            Monitor.Enter(messages);
            ParamMessage msg = null;
            if (messages.Count > 0)
            {
                msg = messages.Dequeue();
            }
            Monitor.Exit(messages);
            return msg;
        }

    }

}