using System;
namespace XEngine.Editor
{

    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
    public sealed class CFDecoratorAttribute : Attribute
    {
        public readonly Type attributeType;

        public CFDecoratorAttribute(Type attributeType)
        {
            this.attributeType = attributeType;
        }
    }
}